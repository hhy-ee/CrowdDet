import os
import json
import numpy as np
from .image import *
from data.CityPersons import load_json_lines

PERSON_CLASSES = ['background', 'person']
# DBBase
class Database(object):
    def __init__(self, gtpath=None, dtpath=None, body_key=None, head_key=None, mode=0):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.eval_mode = mode
        self.loadgtData(gtpath, body_key, head_key, mode)
        self.loaddtData(dtpath, body_key, head_key)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    def loadgtData(self, fpath, body_key=None, head_key=None, mode=None):
        if 'CityPersons' in fpath:
            records = load_json_lines(fpath, False, mode)
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, ['background', 'pedestrian'], True)
        else:
            assert os.path.isfile(fpath), fpath + " does not exist!"
            with open(fpath, "r") as f:
                lines = f.readlines()
            records = [json.loads(line.strip('\n')) for line in lines]
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, ['background', 'person'], True)

    def loaddtData(self, fpath, body_key=None, head_key=None):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        with open(fpath, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        for record in records:
            self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False)
            self.images[record["ID"]].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2"):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst)-1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = (self._gtNum - self._ignNum) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            missrate = 1.0 - recall
            fppi = fp / self._imageNum
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        total_det = len(self.scorelist)
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp/total_images)

        AP = _calculate_map(rpX, rpY)
        return AP, (rpX, rpY, thr, fpn, recalln, fppi)


class MyDatabase(object):
    def __init__(self, mode, gtpath=None, dtpath=None, body_key=None, head_key=None, len_data=None):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.len_data = len_data
        self.eval_mode = mode
        self.loadData(gtpath, body_key, head_key, True)
        self.loadData(dtpath, body_key, head_key, False)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        with open(fpath, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        if if_gt:
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
                if len(self.images) == self.len_data:
                    break
        else:
            for record in records:
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False)
                self.images[record["ID"]].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        pltscrlist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].my_compare_caltech(thres)
            scorelist.extend(result)
            result.sort(key=lambda x: x[0][4], reverse=True)
            pltscrlist.append(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][4], reverse=True)
        self.scorelist = scorelist
        return pltscrlist

    def eval_MR(self, ref="CALTECH_-2"):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst)-1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = (self._gtNum - self._ignNum) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            missrate = 1.0 - recall
            fppi = fp / self._imageNum
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        total_det = len(self.scorelist)
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp/total_images)

        AP = _calculate_map(rpX, rpY)
        return AP, (rpX, rpY, thr, fpn, recalln, fppi)


class InferenceDatabase(object):
    def __init__(self, gtpath=None, img_path=None, body_key=None, mode=0):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.eval_mode = mode
        self.loadData(gtpath, img_path, body_key, None)
        self.scorelist = None

    def loadData(self, fpath, ipath, body_key=None, head_key=None):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        with open(fpath, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]

        for record in records:
            if record["ID"] == ipath.split('/')[-1].split('.')[0]:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
                break

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist
        return scorelist

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dt_matched = np.zeros(len(dtboxes))
        gt_matched = np.zeros(len(gtboxes))

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes, True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes, False)
        else:
            return list()

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist