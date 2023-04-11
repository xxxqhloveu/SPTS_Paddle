import os
import time
import copy
import glob
import json
import numpy as np
import editdistance as ed

from tqdm import tqdm
from shapely.geometry import Point, LineString

import cv2

import paddle

# from utils.misc import decode_seq
# from utils.visualize import visualize_decoded_result

@paddle.no_grad()
def validate(model, dataloader, epoch, config, logger):
    model.eval()
    output_folder = os.path.join(config['Global']['save_model_dir'], 'results', f'ep{epoch:03d}')

    eval_run_cost = 0.0 # model运行eval跑出结果的时间
    results = []
    for data in tqdm(dataloader):
        # seq = data['val_sequence']
        eval_start = time.time()
        # output, prob = model(samples, seq)

        # from test_forward import make_data
        # image, mask, seq = make_data()
        # data = {}
        # data['image'] = paddle.to_tensor(np.expand_dims(image, axis=0).astype("float32"))
        # data['mask'] = paddle.to_tensor(np.expand_dims(mask, axis=0).astype("float32"))
        # data['sequence'] = paddle.to_tensor(seq.astype("int64"))
        
        output, prob = model(data)
        output = output[0].numpy()
        prob = prob[0].numpy()
        result = decode_pred_seq(output, prob, data['target'], config)
        eval_run_cost += time.time() - eval_start
        results.extend(result)

        if config['Eval']['visualize']:
            image = cv2.imread(os.path.join(config['Eval']['dataset']['data_dir'], 
                                            config['Eval']['dataset']['label_file_list'][0]))
            image = visualize_decoded_result(image, result)
            save_path = os.path.join(output_folder, 'vis', "img_"+str(data['target']['image_id'])+".jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

    json_path = os.path.join(output_folder, config['Eval']['dataset']['label_file_list'][1])
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        f.write(json.dumps(results, indent=4))

    gt_folder = config['Eval']['rec']['gt_folder']
    IS_WORDSPOTTING = config['Eval']['rec']['IS_WORDSPOTTING']
    lexicon_paths = config['Eval']['rec']['lexicon_paths']
    pair_paths = config['Eval']['rec']['pair_paths']
    lexicon_type = config['Eval']['rec']['lexicon_type']

    read_start = time.time()
    if config['Eval']['rec']['with_lexicon']:
        lexicon_path = lexicon_paths[lexicon_type]
        pair_path = pair_paths[lexicon_type]
        lexicons = read_lexicon(lexicon_path)
        pairs = read_pair(pair_path)
    else:
        lexicons = None
        pairs = None

    print('Reading GT')
    gts = read_gt(gt_folder, IS_WORDSPOTTING)
    print('Reading and Processing Results')
    results = read_result(results, lexicons, pairs, 0.4, gt_folder, lexicon_type)
    read_cost = time.time() - read_start

    print('Evaluating')
    conf_thres_list = np.arange(0.8, 0.95, 0.01)
    hmeans = []
    recalls = []
    precisions = []
    compute_result_time = 0.0  # 计算result准确性时间
    for conf_thres in conf_thres_list:
        result_time = time.time()
        precision, recall, hmean, pgt, ngt, ndet = evaluate(
            results=results,
            gts=gts,
            conf_thres=conf_thres,
        )
        compute_result_time += time.time() - result_time
        hmeans.append(hmean); recalls.append(recall); precisions.append(precision)

    max_hmean = max(hmeans)
    max_hmean_index = len(hmeans) - hmeans[::-1].index(max_hmean) - 1
    precision = precisions[max_hmean_index]
    recall = recalls[max_hmean_index]
    conf_thres = conf_thres_list[max_hmean_index]
    msg = "[Epoch {}/{}] Precision: {:.4f}, Recall: {:.4f}, Hmean: {:.4f}, Conf Thres: {:.4f}, Eval cost:{:4f}, Compute result cost:{:4f}, ReadGT cost:{:4f}".format(
                                            epoch, 
                                            config['Global']['epoch_num'], 
                                            precision,
                                            recall,
                                            max_hmean,
                                            conf_thres,
                                            eval_run_cost,
                                            compute_result_time,
                                            read_cost)
    logger.info(msg)
    # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Hmean: {max_hmean:.4f}, Conf Thres: {conf_thres:.4f}')

def read_gt(gt_folder, IS_WORDSPOTTING):
    gts = glob.glob(f"{gt_folder}/*.txt")
    gts.sort()

    gt_dict = {}
    for i in gts:
        lines = open(i, "r").readlines()
        imid = int(os.path.basename(i)[:-4])
        points = []
        recs = []
        dontcares = []
        for line in lines:
            if not line: 
                continue

            line_split = line.strip().split(",####")

            dontcare = False
            rec = line_split[1]
            if rec == "###":
                dontcare = True
            else:
                if IS_WORDSPOTTING:
                    if include_in_dictionary(rec) == False: 
                        dontcare = True
                    else:
                        rec = include_in_dictionary_transcription(rec)

            coords = line_split[0]
            coords = coords.split(",")
            coords = [int(ele) for ele in coords]
            center_pt = poly_center(coords)
            center_pt = Point(center_pt[0], center_pt[1])

            points.append(center_pt)
            recs.append(rec)
            dontcares.append(dontcare)
            matched = [0] * len(recs)

        gt_dict[imid] = [points, recs, matched, dontcares]

    return gt_dict

def decode_seq(seq, type, config):
    seq = seq[seq != config['Loss']['padding_index']]
    if type == 'input':
        seq = seq[1:]
    elif type == 'output':
        seq = seq[:-1]
    elif type == 'none':
        seq = seq 
    else:
        raise ValueError
    seq = seq.reshape([-1, 27])

    decode_result = []
    for text_ins_seq in seq:
        point_x = text_ins_seq[0] / config['Global']['num_bins']
        point_y = text_ins_seq[1] / config['Global']['num_bins']
        recog = []
        for index in text_ins_seq[2:]:
            if index == config['Global']['recog_pad_index']:
                break 
            if index == config['Global']['recog_pad_index'] - 1:
                continue
            recog.append(config['Global']['char'][index - config['Global']['num_bins']])
        recog = ''.join(recog)
        decode_result.append(
            {'point': (point_x.item(), point_y.item()), 'recog': recog}
        )
    return decode_result

def decode_pred_seq(index_seq, prob_seq, target, config):
    index_seq = index_seq[:-1]
    prob_seq = prob_seq[:-1]
    # if len(index_seq) % 27 != 0:
    #     index_seq = index_seq[:-len(index_seq)%27]
    #     prob_seq = index_seq[:-len(index_seq)%27]
    if index_seq.shape[0] % 27 != 0:
        index_seq = index_seq[:-index_seq.shape[0]%27]
        prob_seq = index_seq[:-index_seq.shape[0]%27]

    decode_results = decode_seq(index_seq, 'none', config)
    # confs = paddle.reshape(prob_seq, [-1, 27]).mean(-1)
    confs = prob_seq.reshape([-1, 27]).mean(-1)

    image_id = int(target['image_id'])
    image_h, image_w = target['size'][-1].numpy()
    results = []
    for decode_result, conf in zip(decode_results, confs):
        point_x = decode_result['point'][0] * image_w 
        point_y = decode_result['point'][1] * image_h 
        recog = decode_result['recog']
        result = {
            'image_id': image_id,
            'polys': [[point_x.item(), point_y.item()]],
            'rec': recog,
            'score': conf.item()
        }
        results.append(result)

    return results

def find_match_word(rec_str, lexicon, pair):
    rec_str = rec_str.upper()
    match_word = ''
    match_dist = 100
    for word in lexicon:
        word = word.upper()
        ed_dist = ed.eval(rec_str, word)
        norm_ed_dist = ed_dist / max(len(word), len(rec_str))
        if norm_ed_dist < match_dist:
            match_dist = norm_ed_dist
            if pair:
                match_word = pair[word]
            else:
                match_word = word
    return match_word, match_dist

def read_lexicon(lexicon_path):
    if lexicon_path.endswith('.txt'):
        lexicon = open(lexicon_path, 'r').read().splitlines()
        lexicon = [ele.strip() for ele in lexicon]
    else:
        lexicon = {}
        lexicon_dir = os.path.dirname(lexicon_path)
        num_file = len(os.listdir(lexicon_dir))
        assert(num_file % 2 == 0)
        for i in range(num_file // 2):
            lexicon_path_ = lexicon_path + f'{i+1:d}.txt'
            lexicon[i] = read_lexicon(lexicon_path_)
    return lexicon

def read_pair(pair_path):
    if 'ctw1500' in pair_path:
        return None

    if pair_path.endswith('.txt'):
        pair_lines = open(pair_path, 'r').read().splitlines()
        pair = {}
        for line in pair_lines:
            line = line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word)+1:]
            pair[word] = word_gt
    else:
        pair = {}
        pair_dir = os.path.dirname(pair_path)
        num_file = len(os.listdir(pair_dir))
        assert(num_file % 2 == 0)
        for i in range(num_file // 2):
            pair_path_ = pair_path + f'{i+1:d}.txt'
            pair[i] = read_pair(pair_path_)
    return pair    

def poly_center(poly_pts):
    poly_pts = np.array(poly_pts).reshape(-1, 2)
    num_points = poly_pts.shape[0]
    line1 = LineString(poly_pts[int(num_points/2):])
    line2 = LineString(poly_pts[:int(num_points/2)])
    mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
    mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
    return (mid_pt1 + mid_pt2) / 2

### official code
def include_in_dictionary(transcription):
    #special case 's at final
    if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
        transcription = transcription[0:len(transcription)-2]
    #hypens at init or final of the word
    transcription = transcription.strip('-');
    specialCharacters = str("'!?.:,*\"()·[]/");
    for character in specialCharacters:
        transcription = transcription.replace(character,' ')
    transcription = transcription.strip()
    if len(transcription) != len(transcription.replace(" ","")) :
        return False;
    if len(transcription) < 3:
        return False;
    notAllowed = str("×÷·");
    range1 = [ ord(u'a'), ord(u'z') ]
    range2 = [ ord(u'A'), ord(u'Z') ]
    range3 = [ ord(u'À'), ord(u'ƿ') ]
    range4 = [ ord(u'Ǆ'), ord(u'ɿ') ]
    range5 = [ ord(u'Ά'), ord(u'Ͽ') ]
    range6 = [ ord(u'-'), ord(u'-') ]
    for char in transcription :
        charCode = ord(char)
        if(notAllowed.find(char) != -1):
            return False
        valid = ( charCode>=range1[0] and charCode<=range1[1] ) or ( charCode>=range2[0] and charCode<=range2[1] ) or ( charCode>=range3[0] and charCode<=range3[1] ) or ( charCode>=range4[0] and charCode<=range4[1] ) or ( charCode>=range5[0] and charCode<=range5[1] ) or ( charCode>=range6[0] and charCode<=range6[1] )
        if valid == False:
            return False
    return True
    
def include_in_dictionary_transcription(transcription):
    #special case 's at final
    if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
        transcription = transcription[0:len(transcription)-2]
    #hypens at init or final of the word
    transcription = transcription.strip('-');            
    specialCharacters = str("'!?.:,*\"()·[]/");
    for character in specialCharacters:
        transcription = transcription.replace(character,' ')
    transcription = transcription.strip()
    return transcription

# def read_result(result_path, lexicons, pairs, match_dist_thres, gt_folder, lexicon_type):
#     results = json.load(open(result_path, 'r'))
def read_result(results, lexicons, pairs, match_dist_thres, gt_folder, lexicon_type):
    results.sort(reverse=True, key=lambda x: x['score'])

    results = [result for result in results if len(result['rec']) > 0]

    if not lexicons is None:
        print('Processing Results using Lexicon')
        new_results = []
        for result in tqdm(results):
            rec = result['rec']
            if lexicon_type == 2:
                lexicon = lexicons[result['image_id'] - 1]
                pair = pairs[result['image_id'] - 1]
            else:
                lexicon = lexicons
                pair = pairs

            match_word, match_dist = find_match_word(rec, lexicon, pair)
            if match_dist < match_dist_thres or \
               (('gt_ic13' in gt_folder or 'gt_ic15' in gt_folder) and lexicon_type == 0):
                rec = match_word
            else:
                continue
            result['rec'] = rec
            new_results.append(result)
        results = new_results

    return results

def evaluate(results, gts, conf_thres):

    gts = copy.deepcopy(gts)
    results = copy.deepcopy(results)

    ngt = sum([len(ele[0]) for ele in gts.values()])
    ngt -= sum([sum(ele[3]) for ele in gts.values()])

    ndet = 0; ntp = 0
    for result in results:
        confidence = result["score"]
        if confidence < conf_thres:
            continue

        image_id = result['image_id']
        pred_coords = result["polys"]
        pred_rec = result["rec"]
        pred_point = Point(pred_coords[0][0], pred_coords[0][1])

        gt_imid = gts[image_id]
        gt_points = gt_imid[0]
        gt_recs = gt_imid[1]
        gt_matched = gt_imid[2]
        gt_dontcare = gt_imid[3]

        dists = [pred_point.distance(gt_point) for gt_point in gt_points]
        minvalue = min(dists)
        idxmin = dists.index(minvalue)
        if gt_recs[idxmin] == "###" or gt_dontcare[idxmin] == True:
            continue
        if pred_rec.upper() == gt_recs[idxmin].upper() and gt_matched[idxmin] == 0:
            gt_matched[idxmin] = 1
            ntp += 1

        ndet += 1

    if ndet == 0 or ntp == 0:
        recall = 0; precision = 0; hmean = 0
    else:
        recall = ntp / ngt
        precision  = ntp / ndet
        hmean = 2 * recall * precision / (recall + precision)
    return precision, recall, hmean, ntp, ngt, ndet
