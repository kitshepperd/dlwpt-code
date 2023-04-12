import SimpleITK as sitk
import glob
import os
import numpy as np
import tqdm
import warnings

if __name__=="__main__":



    mhd_list = glob.glob('../../data-unversioned/part2/luna/subset*/*.mhd')
    series_uids = np.array([os.path.split(p)[-1][:-4] for p in mhd_list])
    series_uids = series_uids[[295, 336]]
    print(series_uids)
    for series_uid in series_uids:
        mhd_path = glob.glob(
            '../../data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )
        print(mhd_path)
        ct_mhd = sitk.ReadImage(mhd_path)

