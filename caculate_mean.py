import pandas as pd
import os
import numpy as np
import re


def cal_mean(path):
    files = os.listdir(path)

    book = np.zeros(50)
    book = {}
    start = 65535
    end = -1
    for f in files:
        # index = re.findall(r'InferenceDirectly_(\d+?)\.xlsx', f)[0]
        index = re.findall(r'InferenceDirectly_(\d+?)_\d+\.xlsx', f)[0]
        index = int(index)
        if index < start:
            start = index
        if index > end:
            end = index
        xlsx_path = os.path.join(path, f)
        temp_df = pd.read_excel(xlsx_path)
        res = np.mean(temp_df['ID_HGNN_loss'].values)

        if not book.get(index):
            book[index] = [res]
        else:
            book[index].append(res)
        # print(index, res)
    tot = 0
    for i in range(start, end + 1):
        print(i, book[i])
        tot += sum(book[i])
    print('mean', tot / len(files))
    return tot / len(files)


if __name__ == '__main__':
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round11"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round13"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round14"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round16"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round17"  # *
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round18"  # *
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round19"  # x  server 为 2 的时候误差非常大
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round20"  # x  server 为 2 的时候误差非常大
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round21"  # s = 2 loss = 10  but s = 8 loss = 11
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round22" # seed = 2023
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round23" # seed = 2023 best
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round24"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round25"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round28"  # *
    path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round30"  # *  只有 server = 3 时, 延迟达到 6
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round32"  ## *
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round35"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round36"
    path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round37"  # 均值最小
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round42"
    # path = r"E:\wyh\论文\UNIC\LOGNN\data\multi_scales_test\small\round43"

    files = os.listdir(path)

    book = np.zeros(50)
    start = 65535
    end = -1
    for f in files:
        index = re.findall(r'InferenceDirectly_(\d+?)\.xlsx', f)[0]
        index = int(index)
        if index < start:
            start = index
        if index > end:
            end = index
        xlsx_path = os.path.join(path, f)
        temp_df = pd.read_excel(xlsx_path)
        res = np.mean(temp_df['ID_HGNN_loss'].values)
        book[index] = res
        # print(index, res)
    tot = 0
    for i in range(start, end + 1):
        print(i, book[i])
        tot += book[i]
    print('mean', tot / len(files))
# round 11 E:\wyh\论文\UNIC\LOGNN\TO_models\Pretrain\small\2023-11-26\round2\HGNN_2048_9607_5.0715888076358375.pt