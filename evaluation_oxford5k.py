# oxford_5k_evaluation/evaluation_oxford5k.py
import os
import sys
import time
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# --- 导入你的系统模块 ---
from feature_extractor import (
    extract_sift_features,
    extract_descriptors_for_database,
)
from vocabulary_builder import build_vocabulary
from indexer import create_tfidf_index
from retriever import retrieve_similar_images
from utils import list_image_files, save_model, load_model

# --- 评估配置 ---
OXFORD_DATA_DIR = "oxford5k"  # 包含 images 和 groundtruth.json 的目录路径
OXFORD_IMAGE_DIR = os.path.join(OXFORD_DATA_DIR, "images")
GROUNDTRUTH_JSON_PATH = os.path.join(OXFORD_DATA_DIR, "groundtruth.json")
MODELS_DIR_EVAL = f"results/models_oxford_eval"  # 评估模型的存储路径

# --- 系统参数 ---
PARAMS = {
    "NUM_VISUAL_WORDS": 1000,  # 论文中在5K数据集上使用1M达到峰值，可尝试1k, 10k, 50k TODO
    "ENABLE_SPATIAL_RERANKING": True,  # 是否启用空间重排序
    "TOP_N_INITIAL_BOW_CANDIDATES": 200,  # BoW 初步检索候选数量 (用于空间重排序)
    "NUM_TO_SPATIALLY_RERANK": 100,  # 实际进行空间验证的候选数量 TODO
    "SPATIAL_MIN_INLIERS": 4,  # 空间验证所需的最小内点数
    "SPATIAL_RANSAC_REPROJ_THRESH": 5.0,  # RANSAC 重投影阈值
    "SPATIAL_RATIO_THRESH": 0.75,  # SIFT 特征匹配时的比率检验阈值
}

VOCAB_PATH_EVAL = os.path.join(
    MODELS_DIR_EVAL, f"vocab_{PARAMS['VOCAB_BUILDER']}_{PARAMS['NUM_VISUAL_WORDS']}.pkl"
)
TFIDF_MATRIX_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "tfidf_matrix.pkl")
TFIDF_TRANSFORMER_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "tfidf_transformer.pkl")
DB_IMAGE_PATHS_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "db_image_paths.pkl")


def load_ground_truth(json_path, image_base_dir):
    """
    加载 groundtruth.json 文件。
    返回一个查询信息列表，每个元素是一个字典，包含：
    'landmark_name', 'query_image_path', 'positive_paths', 'junk_paths'
    """
    if not os.path.exists(json_path):
        print(f"错误: Ground truth 文件未找到于 {json_path}")
        return []

    with open(json_path, "r") as f:
        gt_data = json.load(f)

    queries_data = []
    for landmark_name, landmark_info in gt_data.items():
        query_filenames = landmark_info.get("query", [])
        good_filenames = landmark_info.get("good", [])
        ok_filenames = landmark_info.get("ok", [])
        junk_filenames = landmark_info.get("junk", [])

        positive_filenames_set = set(good_filenames).union(set(ok_filenames))
        junk_filenames_set = set(junk_filenames)

        for query_filename in query_filenames:
            query_image_path = os.path.join(image_base_dir, query_filename)
            if not os.path.exists(query_image_path):
                print(
                    f"警告: 查询图片 {query_filename} 未在目录 {image_base_dir} 中找到，跳过此查询。"
                )
                continue

            current_positive_paths = {
                os.path.join(image_base_dir, f)
                for f in positive_filenames_set
                if os.path.exists(os.path.join(image_base_dir, f))
            }
            current_junk_paths = {
                os.path.join(image_base_dir, f)
                for f in junk_filenames_set
                if os.path.exists(os.path.join(image_base_dir, f))
            }

            # 从正样本中移除查询图像本身，因为它不应作为自己的检索结果被计分
            current_positive_paths.discard(query_image_path)

            queries_data.append(
                {
                    "landmark_name": landmark_name,
                    "query_image_path": query_image_path,
                    "positive_paths": current_positive_paths,
                    "junk_paths": current_junk_paths,
                }
            )

    print(f"从 {json_path} 加载了 {len(queries_data)} 个查询。")
    return queries_data


def calculate_ap(ranked_list_paths, positive_set_paths, junk_set_paths=None):
    """
    计算平均精度 (Average Precision - AP)。
    ranked_list_paths: 系统返回的排序后的图像路径列表。
    positive_set_paths: 此查询的相关的（正样本）图像路径集合。
    junk_set_paths: 此查询的“忽略”的（junk）图像路径集合。
    """
    if junk_set_paths is None:
        junk_set_paths = set()

    # 根据论文方法，从排序列表中过滤掉 junk 图像
    processed_ranked_list = [p for p in ranked_list_paths if p not in junk_set_paths]

    if not positive_set_paths or not processed_ranked_list:
        return 0.0  # 如果没有正样本或处理后的排序列表为空，AP为0

    num_relevant_docs = len(positive_set_paths)
    retrieved_relevant_count = 0
    precision_sum = 0.0

    for i, path in enumerate(processed_ranked_list):
        if path in positive_set_paths:
            retrieved_relevant_count += 1
            precision_at_k = retrieved_relevant_count / (i + 1)
            precision_sum += precision_at_k

    if retrieved_relevant_count == 0:  # 没有检索到任何相关的文档
        return 0.0

    # AP 标准定义: sum of (precision@k * is_relevant@k) / total_relevant_documents
    # 由于我们只在检索到相关文档时累加 precision@k (此时 is_relevant@k = 1),
    # 所以可以直接用 precision_sum / num_relevant_docs
    return precision_sum / num_relevant_docs


def main():
    os.makedirs(MODELS_DIR_EVAL, exist_ok=True)

    print("--- 系统参数 ---")
    for key, value in PARAMS.items():
        print(f"{key}: {value}")
    print("-------------------------")

    # 1. 数据准备 (加载 ground truth)
    print("\n--- 加载 Oxford 5K Ground Truth ---")
    if not os.path.exists(OXFORD_IMAGE_DIR):
        print(f"错误: 图像目录未找到于 {OXFORD_IMAGE_DIR}")
        print("请确保已下载数据集并将图像放入该目录。")
        return

    queries = load_ground_truth(GROUNDTRUTH_JSON_PATH, OXFORD_IMAGE_DIR)
    if not queries:
        print("加载查询失败。退出。")
        return

    all_oxford_image_paths = sorted(list_image_files(OXFORD_IMAGE_DIR))
    if not all_oxford_image_paths:
        print(f"在 {OXFORD_IMAGE_DIR} 中未找到图像。退出。")
        return
    print(f"用于索引的 Oxford 5K 数据集图像总数: {len(all_oxford_image_paths)}")

    # 2. 构建/加载词汇表和索引 (这些是系统全局的，构建一次)
    vocabulary = None
    tfidf_matrix_db = None
    tfidf_transformer_db = None
    db_image_paths_indexed = None

    build_time_start = time.time()
    # 尝试加载预构建的模型
    if (
        os.path.exists(VOCAB_PATH_EVAL)
        and os.path.exists(TFIDF_MATRIX_PATH_EVAL)
        and os.path.exists(TFIDF_TRANSFORMER_PATH_EVAL)
        and os.path.exists(DB_IMAGE_PATHS_PATH_EVAL)
    ):
        print("\n--- 加载预构建的评估模型 ---")
        vocabulary = load_model(VOCAB_PATH_EVAL)
        tfidf_matrix_db = load_model(TFIDF_MATRIX_PATH_EVAL)
        tfidf_transformer_db = load_model(TFIDF_TRANSFORMER_PATH_EVAL)
        db_image_paths_indexed = load_model(DB_IMAGE_PATHS_PATH_EVAL)
        if not all(
            [
                vocabulary,
                tfidf_matrix_db is not None,
                tfidf_transformer_db,
                db_image_paths_indexed,
            ]
        ):
            print("一个或多个模型加载失败。需要重新构建。")
            (
                vocabulary,
                tfidf_matrix_db,
                tfidf_transformer_db,
                db_image_paths_indexed,
            ) = (
                None,
                None,
                None,
                None,
            )  # 强制重建
        else:
            print("所有模型加载成功。")

    if vocabulary is None:  # 如果未加载或加载失败，则重新构建
        print("\n--- 为 Oxford 5K 构建词汇表和索引 ---")
        print("提取数据库描述符用于构建词汇表...")
        all_db_descriptors, _ = extract_descriptors_for_database(all_oxford_image_paths)
        if all_db_descriptors is None:
            print("提取描述符失败。退出。")
            return

        print(
            f"使用 MiniBatchKMeans 构建包含 {PARAMS['NUM_VISUAL_WORDS']} 个词的词汇表..."
        )
        vocabulary = build_vocabulary(
            all_db_descriptors, PARAMS["NUM_VISUAL_WORDS"], random_state=42
        )

        if vocabulary is None:
            print("构建词汇表失败。退出。")
            return
        save_model(vocabulary, VOCAB_PATH_EVAL)

        print("创建 TF-IDF 索引...")
        tfidf_matrix_db, tfidf_transformer_db, db_image_paths_indexed = (
            create_tfidf_index(
                all_oxford_image_paths,
                vocabulary,
                extract_sift_features,  # 查询时使用全图特征
            )
        )
        if tfidf_matrix_db is None:
            print("创建 TF-IDF 索引失败。退出。")
            return
        save_model(tfidf_matrix_db, TFIDF_MATRIX_PATH_EVAL)
        save_model(tfidf_transformer_db, TFIDF_TRANSFORMER_PATH_EVAL)
        save_model(db_image_paths_indexed, DB_IMAGE_PATHS_PATH_EVAL)

    build_time_end = time.time()
    total_build_time = build_time_end - build_time_start
    print(f"词汇表和索引构建时间: {total_build_time:.2f} 秒")

    # 3. 执行查询和评估
    print("\n--- 开始评估查询 ---")
    all_ap_scores = []
    total_query_time = 0
    num_processed_queries = 0

    for i, query_info in tqdm(
        enumerate(queries), desc="Processing Queries", total=len(queries)
    ):
        print(
            f"处理查询 {i+1}/{len(queries)}: 地标 '{query_info['landmark_name']}', 图片 '{os.path.basename(query_info['query_image_path'])}'..."
        )

        # 查询特征由 retriever 内部从整个 query_image_path 提取
        query_time_start = time.time()

        # retrieve_similar_images 应返回完整的排序列表以计算 AP
        # db_image_paths_indexed 是与 tfidf_matrix_db 中的行对应的路径列表
        ranked_results_paths, _ = retrieve_similar_images(
            query_image_path=query_info["query_image_path"],
            feature_extractor_func=extract_sift_features,  # 用于空间验证中的候选图像
            vocabulary_model=vocabulary,
            tfidf_transformer=tfidf_transformer_db,  # 包含IDF分数，用于空间重排序
            database_tfidf_matrix=tfidf_matrix_db,
            database_image_paths=db_image_paths_indexed,  # 索引中所有图像的路径
            top_n_initial_bow=len(db_image_paths_indexed),  # 为AP计算获取完整排序
            top_n_final=len(db_image_paths_indexed),  # 为AP计算获取完整排序
            enable_spatial_reranking=PARAMS["ENABLE_SPATIAL_RERANKING"],
            num_to_rerank_spatial=PARAMS["NUM_TO_SPATIALLY_RERANK"],
            # spatial_verifier.py 中的参数通过 retriever.py 内部调用时传递，或在PARAMS中设置
            # retriever.py 中的 retrieve_similar_images 可能需要更新以传递这些参数
        )
        query_time_end = time.time()
        total_query_time += query_time_end - query_time_start
        num_processed_queries += 1

        current_ap = calculate_ap(
            ranked_results_paths, query_info["positive_paths"], query_info["junk_paths"]
        )
        all_ap_scores.append(current_ap)
        print(
            f"  查询 {os.path.basename(query_info['query_image_path'])} AP: {current_ap:.4f}"
        )

    # 4. 报告结果
    mean_ap = np.mean(all_ap_scores) if all_ap_scores else 0.0
    avg_query_time = (
        total_query_time / num_processed_queries if num_processed_queries > 0 else 0.0
    )

    print("\n--- 评估总结 ---")
    print(f"数据集: Oxford 5K")
    print(f"处理的查询数量: {num_processed_queries}/{len(queries)}")
    print(f"平均精度均值 (mAP): {mean_ap:.4f}")
    print(f"总计词汇表和索引构建时间: {total_build_time:.2f} 秒")
    print(f"平均查询时间: {avg_query_time:.3f} 秒/查询")

    print("\n--- 使用的系统参数 ---")
    for key, value in PARAMS.items():
        print(f"{key}: {value}")

    results_summary = {
        "dataset": "Oxford 5K",
        "mAP": mean_ap,
        "num_queries_processed": num_processed_queries,
        "total_queries": len(queries),
        "build_time_seconds": total_build_time,
        "avg_query_time_seconds": avg_query_time,
        "parameters": PARAMS,
    }
    results_filename = f"results_oxford5k_{PARAMS['VOCAB_BUILDER']}_{PARAMS['NUM_VISUAL_WORDS']}_spatial_{PARAMS['ENABLE_SPATIAL_RERANKING']}.json"
    results_filepath = os.path.join(MODELS_DIR_EVAL, results_filename)
    with open(results_filepath, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    print(f"\n结果概要已保存到: {results_filepath}")


if __name__ == "__main__":
    main()
