## 系统部署和使用

### 环境要求

```
# requirements.txt
gradio==5.31.0
numpy==1.26.4
opencv_contrib_python==4.11.0.86
opencv_python==4.11.0.86
Pillow==11.1.0
scikit_learn==1.6.1
tqdm==4.67.1
```

### 目录结构

```
image_retrieval_system/
├── app.py                    # 主应用程序
├── feature_extractor.py      # 特征提取模块
├── vocabulary_builder.py     # 词汇构建模块
├── indexer.py               # 索引构建模块
├── retriever.py             # 检索核心模块
├── spatial_verifier.py      # 空间验证模块
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖包列表
├── database_images/         # 数据库图像目录
|   ├── test01/
|   │   ├── img01.jpg
|   │   ├── img02.jpg
|   │   └── ...
|   ├── test02/
|   │   └── ...
├── models/                  # 模型存储目录
│   └── test01/
|   │   ├── vocabulary.pkl       # 词汇表模型
│   │   ├── tfidf_matrix.pkl    # TF-IDF矩阵
│   │   └── ...
│   ├── test02/
│   │   └── ...
└── query_images/           # 查询图像目录（可选）
```

### 运行步骤

1. 安装依赖包

    ```bash
    cd image_retrieval_system
    python -m venv venv
    ## Linux
    source venv/bin/activate
    ## Windows
    venv/Scriptsactivate

    pip install -r requirements.txt
    ```

2. 准备数据集
   在`database_images`目录下创建子目录放置你的图像数据集，每个子目录代表一个数据集。
3. 启动系统
    ```bash
    python app.py
    ```
    随后在浏览器中打开终端输出的 URL，即可访问系统界面。
    ![系统界面](docs/images/system_interface.png)

## Oxford5K 数据集评估

1. 下载数据集
   我们使用 kaggle 上整理的 Oxford5K 数据集，包含 images 子目录和 groundtruth.json 文件，前者包含所有图像，后者包含每个地标的形式的标签：

    ```json
    {
        'landmark': {
            'ok': [list of images],
            'good': [list of images],
            'junk': [list of images],
            'query': [list of images]
        }
    }
    ```

2. 运行评估脚本
   在`evaluation_oxford5k.py`中，首先修改`OXFORD_DATA_DIR`和`MODELS_DIR_EVAL`为数据集路径和结果存储路径，随后调整有关参数（主要是`NUM_VISUAL_WORDS`和`ENABLE_SPATIAL_RERANKING`等），然后运行脚本：

```bash
python evaluation_oxford5k.py
```

3. 结果查看
    - 评估结果将保存在 `results/` 目录下
    - 包含以下信息：
        - mAP (平均精度均值)
        - 查询处理时间统计
        - 索引构建时间
        - 详细的评估参数配置
