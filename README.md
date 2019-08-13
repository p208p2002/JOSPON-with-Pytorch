# JOSPON-with-Pytorch
Judge of sentence positive or negative with PyTorch
句子正負面情緒判斷類神經網路模型，使用PyTorch
> PyTorch練習小專案

[資料集來源(賣外評論資料)](https://github.com/SophonPlus/ChineseNlpCorpus)

# 模型定義
- 包含兩個隱藏層
- hidden、hidden2 使用 `relu`
- out 使用 `sigmoid`
```
(hidden): Linear(in_features=200, out_features=150, bias=True)
(hidden2): Linear(in_features=150, out_features=100, bias=True)
(out): Linear(in_features=100, out_features=1, bias=True)
```

# 檔案說明
| 名稱             | 說明                                   |
|------------------|----------------------------------------|
| demo.py          | 模型示範                               |
| jwp_train_bce.py | 模型訓練                               |
| JWP.py           | 神經網路定義類                         |
| make_w2v_set.py  | 製作訓練所需的檔案(資料集文字轉換向量) |
| w2v_seg.py       | 製作w2v訓練資料(斷詞)                  |
| w2v_train.py     | 訓練w2v模型                            |
| w2v_test.py      | 測試文字在w2v模型的向量                |
| W2V_SV.py        | 轉換句到句向量類                       |

# 建置流程
1. 先處理wiki data，準備製作w2v模型
    - 使用檔案:`w2v_seg.py`
2. 使用處理好的wiki data訓練w2v模型
    - 使用檔案:`w2v_train.py`
3. 取得waimai資料集句向量
    - 使用檔案:`make_w2v_set.py`
4. 訓練神經網路
    -  使用檔案:`jwp_train_bce.py`
5. 測試
    - 使用檔案:`demo.py`

> [維基資料集與預訓練模型下載](https://github.com/p208p2002/JOSPON-with-Pytorch/releases)
# 結果
## 訓練結果
```
.
.
.
epoch: 298 loss: 0.556 train_acc: 0.879 test_acc: 0.827 LR: 0.007748409780000004
epoch: 299 loss: 0.556 train_acc: 0.879 test_acc: 0.827 LR: 0.007748409780000004
epoch: 300 loss: 0.556 train_acc: 0.879 test_acc: 0.828 LR: 0.007748409780000004
```
## 測試結果
```
輸入評價:東西好吃
1.0
正面

輸入評價:東西難吃
0.0
反面

輸入評價:速度很快，還不錯吃
1.0
正面

輸入評價:動作慢，不太好吃
0.389
反面
```
