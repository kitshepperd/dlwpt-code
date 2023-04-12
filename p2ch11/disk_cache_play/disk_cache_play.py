from diskcache import FanoutCache, Cache
from util.disk import GzipDisk
import numpy as np
import copy
import csv
import functools
import shutil
import glob
import os
import random
import time



CACHE_PATH = "../../data-unversioned/disk_cache_play/cache"
shutil.rmtree(CACHE_PATH)
os.mkdir(CACHE_PATH)

cache = FanoutCache(CACHE_PATH, disk=GzipDisk)

@cache.memoize(typed=True)
def some_function(num: int):
    return np.ones((50,50,50), dtype=np.float64)*num


for i in range(1000):
    some_function(i)

cache.stats(enable=True)
print(cache.stats())
some_function(500)
some_function(1002)
print(cache.stats())































#
# from collections import namedtuple
#
# import SimpleITK as sitk
# import numpy as np
#
# import torch
# import torch.cuda
# from torch.utils.data import Dataset
# import sqlite3
#
# from util.disk import getCache
# from util.util import XyzTuple, xyz2irc
# from util.logconf import logging
#
# CandidateInfoTuple = namedtuple(
#     'CandidateInfoTuple',
#     'isNodule_bool, diameter_mm, series_uid, center_xyz',
# )
#
#

# DATA_PATH = "../../data-unversioned/part2/luna/subset0"
#
#
#
# class CustomCt:
#     def __init__(self, series_uid):
#         mhd_path = glob.glob(
#             DATA_PATH + '/{}.mhd'.format(series_uid)
#         )[0]
#
#         ct_mhd = sitk.ReadImage(mhd_path)
#         ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
#
#         # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
#         # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
#         # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
#         # The upper bound nukes any weird hotspots and clamps bone down
#         ct_a.clip(-1000, 1000, ct_a)
#
#         self.series_uid = series_uid
#         self.hu_a = ct_a
#
#         self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
#         self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
#         self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
#
#     def getRawCandidate(self, center_xyz, width_irc):
#         center_irc = xyz2irc(
#             center_xyz,
#             self.origin_xyz,
#             self.vxSize_xyz,
#             self.direction_a,
#         )
#
#         slice_list = []
#         for axis, center_val in enumerate(center_irc):
#             start_ndx = int(round(center_val - width_irc[axis]/2))
#             end_ndx = int(start_ndx + width_irc[axis])
#
#             assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
#
#             if start_ndx < 0:
#                 # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
#                 #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
#                 start_ndx = 0
#                 end_ndx = int(width_irc[axis])
#
#             if end_ndx > self.hu_a.shape[axis]:
#                 # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
#                 #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
#                 end_ndx = self.hu_a.shape[axis]
#                 start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
#
#             slice_list.append(slice(start_ndx, end_ndx))
#
#         ct_chunk = self.hu_a[tuple(slice_list)]
#
#         return ct_chunk, center_irc
#
# @functools.lru_cache(1)
# def getCandidateInfoList(requireOnDisk_bool=True):
#     # We construct a set with all series_uids that are present on disk.
#     # This will let us use the data, even if we haven't downloaded all of
#     # the subsets yet.
#     mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
#     presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
#
#     diameter_dict = {}
#     with open('data/part2/luna/annotations.csv', "r") as f:
#         for row in list(csv.reader(f))[1:]:
#             series_uid = row[0]
#             annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
#             annotationDiameter_mm = float(row[4])
#
#             diameter_dict.setdefault(series_uid, []).append(
#                 (annotationCenter_xyz, annotationDiameter_mm),
#             )
#
#     candidateInfo_list = []
#     with open('data/part2/luna/candidates.csv', "r") as f:
#         for row in list(csv.reader(f))[1:]:
#             series_uid = row[0]
#
#             if series_uid not in presentOnDisk_set and requireOnDisk_bool:
#                 continue
#
#             isNodule_bool = bool(int(row[4]))
#             candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
#
#             candidateDiameter_mm = 0.0
#             for annotation_tup in diameter_dict.get(series_uid, []):
#                 annotationCenter_xyz, annotationDiameter_mm = annotation_tup
#                 for i in range(3):
#                     delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
#                     if delta_mm > annotationDiameter_mm / 4:
#                         break
#                 else:
#                     candidateDiameter_mm = annotationDiameter_mm
#                     break
#
#             candidateInfo_list.append(CandidateInfoTuple(
#                 isNodule_bool,
#                 candidateDiameter_mm,
#                 series_uid,
#                 candidateCenter_xyz,
#             ))
#
#     candidateInfo_list.sort(reverse=True)
#     return candidateInfo_list
#
# @functools.lru_cache(1, typed=True)
# def getCt(series_uid):
#     return CustomCt(series_uid)
#
#
# if __name__=="__main__":
#     files = glob.glob(DATA_PATH + "/*")
#     series_uids = [x.split("/")[-1] for x in files]
#     series_uids = [x.split(".")[:-1] for x in series_uids]
#     series_uids = [('.').join(x) for x in series_uids]
#     series_uids = np.unique(np.array(series_uids))
#     getCt(series_uids[0])