# diffuser-inpaint

## 環境構築
```shell
conda env create -n diffuser -f diffuser.yml
conda activate diffuser
```

## 実行
```python
python main.py --input ./dataset/input --mask ./dataset/mask --output ./result
```