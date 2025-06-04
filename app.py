# image_retrieval_system/app.py
from tracemalloc import start
import gradio as gr
import os
import numpy as np
from time import time

from feature_extractor import extract_sift_features, extract_descriptors_for_database
from vocabulary_builder import build_vocabulary
from indexer import create_tfidf_index
from retriever import retrieve_similar_images
from utils import list_image_files, save_model, load_model

# --- Configuration ---
DATABASE_DIR = "database_images"
MODELS_DIR = "models"
VOCAB_PATH = os.path.join(MODELS_DIR, "vocabulary.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
TFIDF_TRANSFORMER_PATH = os.path.join(MODELS_DIR, "tfidf_transformer.pkl")
DB_IMAGE_PATHS_PATH = os.path.join(MODELS_DIR, "db_image_paths.pkl")

NUM_VISUAL_WORDS = 200
TOP_N_FINAL_RESULTS = 5
TOP_N_INITIAL_BOW_CANDIDATES = 50
NUM_TO_SPATIAL_RERANK = 20

# --- Global variables to hold models ---
vocabulary = None
tfidf_matrix_db = None
tfidf_transformer_db = None
database_image_paths = None


def setup_system(force_rebuild=False):
    """Loads models or builds them if they don't exist or force_rebuild is True."""
    global vocabulary, tfidf_matrix_db, tfidf_transformer_db, database_image_paths

    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if (
        not force_rebuild
        and os.path.exists(VOCAB_PATH)
        and os.path.exists(TFIDF_MATRIX_PATH)
        and os.path.exists(TFIDF_TRANSFORMER_PATH)
        and os.path.exists(DB_IMAGE_PATHS_PATH)
    ):

        print("Loading pre-built models...")
        vocabulary = load_model(VOCAB_PATH)
        tfidf_matrix_db = load_model(TFIDF_MATRIX_PATH)
        tfidf_transformer_db = load_model(TFIDF_TRANSFORMER_PATH)
        database_image_paths = load_model(DB_IMAGE_PATHS_PATH)

        if (
            vocabulary
            and tfidf_matrix_db is not None
            and tfidf_transformer_db
            and database_image_paths
        ):
            print("All models loaded successfully.")
            return "Models loaded successfully."
        else:
            print("Failed to load one or more models. Rebuilding...")
            force_rebuild = True  # Force rebuild if loading failed

    if force_rebuild:
        print("Building models from scratch...")
        start_time = time()
        image_paths = list_image_files(DATABASE_DIR)
        if not image_paths:
            msg = f"No images found in {DATABASE_DIR}. Please add images and rebuild."
            print(msg)
            return msg

        print(f"Found {len(image_paths)} images for indexing.")

        # 1. Extract all descriptors from the database
        print("Extracting SIFT descriptors from database images...")
        all_db_descriptors, valid_db_paths_for_vocab = extract_descriptors_for_database(
            image_paths
        )
        if all_db_descriptors is None or len(all_db_descriptors) == 0:
            msg = "No SIFT descriptors could be extracted from database images. Cannot build vocabulary."
            print(msg)
            vocabulary, tfidf_matrix_db, tfidf_transformer_db, database_image_paths = (
                None,
                None,
                None,
                None,
            )
            return msg

        # 2. Build vocabulary
        print("Building visual vocabulary...")
        vocabulary = build_vocabulary(all_db_descriptors, num_clusters=NUM_VISUAL_WORDS)
        if vocabulary is None:
            msg = "Failed to build vocabulary."
            print(msg)
            vocabulary, tfidf_matrix_db, tfidf_transformer_db, database_image_paths = (
                None,
                None,
                None,
                None,
            )
            return msg
        save_model(vocabulary, VOCAB_PATH)

        # 3. Create TF-IDF index
        # We should use only images from which descriptors were successfully extracted for vocab building
        # for consistency, or re-extract for all original valid_image_paths
        print("Creating TF-IDF index...")
        current_image_paths_for_index = list_image_files(
            DATABASE_DIR
        )  # Use all valid images

        tfidf_matrix_db, tfidf_transformer_db, indexed_paths = create_tfidf_index(
            current_image_paths_for_index, vocabulary, extract_sift_features
        )
        if tfidf_matrix_db is None or tfidf_transformer_db is None:
            msg = "Failed to create TF-IDF index."
            print(msg)
            vocabulary, tfidf_matrix_db, tfidf_transformer_db, database_image_paths = (
                None,
                None,
                None,
                None,
            )
            return msg

        database_image_paths = (
            indexed_paths  # Store the paths that were actually indexed
        )

        save_model(tfidf_matrix_db, TFIDF_MATRIX_PATH)
        save_model(tfidf_transformer_db, TFIDF_TRANSFORMER_PATH)
        save_model(database_image_paths, DB_IMAGE_PATHS_PATH)

        end_time = time()
        elapsed_time = end_time - start_time
        msg = f"Models built and saved. Indexed {len(database_image_paths)} images in {elapsed_time:.2f} seconds."
        print(msg)
        return msg

    # This part should ideally not be reached if logic is correct
    return "System setup complete or models loaded."


def search_interface(query_image_path_temp, enable_spatial_reranking_ui):
    """Gradio interface function."""
    if (
        vocabulary is None
        or tfidf_matrix_db is None
        or tfidf_transformer_db is None
        or database_image_paths is None
    ):
        return [
            "System not initialized. Please run setup/rebuild models."
        ], "System not ready."
    if not hasattr(tfidf_transformer_db, "idf_") and enable_spatial_reranking_ui:
        return [
            "TF-IDF transformer incompatible or not loaded correctly (missing idf_). Cannot do spatial re-ranking. Try rebuilding models."
        ], "Model Error."

    if not query_image_path_temp:
        return ["Please upload a query image."], "No query image."

    print(
        f"Querying with image: {query_image_path_temp}, Spatial Re-ranking: {enable_spatial_reranking_ui}"
    )

    result_image_paths, scores_info = retrieve_similar_images(
        query_image_path_temp,
        extract_sift_features,
        vocabulary,
        tfidf_transformer_db,  # Pass the whole transformer
        tfidf_matrix_db,
        database_image_paths,
        top_n_initial_bow=TOP_N_INITIAL_BOW_CANDIDATES,
        top_n_final=TOP_N_FINAL_RESULTS,
        enable_spatial_reranking=enable_spatial_reranking_ui,
        num_to_rerank_spatial=NUM_TO_SPATIAL_RERANK,
    )

    return result_image_paths, scores_info


# --- Initialize system on startup ---
initial_setup_message = setup_system(
    force_rebuild=False
)  # Set to True to force rebuild on first start
print(f"Initial setup message: {initial_setup_message}")


# 06/04 增加数据库和模型选择功能方便实验测试
def get_available_databases():
    """获取DATABASE_DIR下所有可用的数据库子目录"""
    databases = []
    if os.path.exists(DATABASE_DIR):
        for item in os.listdir(DATABASE_DIR):
            item_path = os.path.join(DATABASE_DIR, item)
            if os.path.isdir(item_path):
                databases.append(item)
    return databases


def get_available_models():
    """获取models目录下所有可用的模型子目录"""
    models = []
    models_root = "models"
    if os.path.exists(models_root):
        for item in os.listdir(models_root):
            item_path = os.path.join(models_root, item)
            if os.path.isdir(item_path):
                models.append(item)
    return models


def update_paths_for_database(selected_db, selected_model):
    """根据选择的数据库和模型更新全局路径变量"""
    global DATABASE_DIR, MODELS_DIR, VOCAB_PATH, TFIDF_MATRIX_PATH, TFIDF_TRANSFORMER_PATH, DB_IMAGE_PATHS_PATH

    if selected_db:
        DATABASE_DIR = os.path.join("database_images", selected_db)
    else:
        DATABASE_DIR = "database_images/oxford5k"  # 默认值

    if selected_model:
        MODELS_DIR = os.path.join("models", selected_model)
    else:
        MODELS_DIR = "models/oxford5k_200_AKM"  # 默认值

    # 更新相关路径
    VOCAB_PATH = os.path.join(MODELS_DIR, "vocabulary.pkl")
    TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
    TFIDF_TRANSFORMER_PATH = os.path.join(MODELS_DIR, "tfidf_transformer.pkl")
    DB_IMAGE_PATHS_PATH = os.path.join(MODELS_DIR, "db_image_paths.pkl")

    return f"已切换到数据库: {DATABASE_DIR}, 模型目录: {MODELS_DIR}"


def setup_system_with_params(selected_db, selected_model, force_rebuild=False):
    """根据选择的数据库和模型设置系统"""
    # 更新路径
    path_update_msg = update_paths_for_database(selected_db, selected_model)
    print(path_update_msg)

    # 重置全局变量
    global vocabulary, tfidf_matrix_db, tfidf_transformer_db, database_image_paths
    vocabulary = None
    tfidf_matrix_db = None
    tfidf_transformer_db = None
    database_image_paths = None

    # 调用原有的设置函数
    setup_msg = setup_system(force_rebuild)

    return f"{path_update_msg}\n{setup_msg}"


def search_interface_with_params(
    query_image_path_temp, enable_spatial_reranking_ui, selected_db, selected_model
):
    """扩展的搜索接口函数，支持动态数据库和模型选择"""
    # 先更新路径配置
    update_paths_for_database(selected_db, selected_model)

    # 调用原有的搜索接口
    return search_interface(query_image_path_temp, enable_spatial_reranking_ui)


# --- Gradio Interface ---
with gr.Blocks(title="Image Retrieval System with Spatial Re-ranking") as demo:
    gr.Markdown("# Image Retrieval System with Spatial Re-ranking")
    gr.Markdown(
        "Upload a query image. Spatial re-ranking (inspired by Philbin et al., CVPR 2007) can be enabled."
    )

    # 资源选择区域
    with gr.Row():
        gr.Markdown("## 资源配置")

    with gr.Row():
        available_dbs = get_available_databases()
        db_dropdown = gr.Dropdown(
            choices=available_dbs,
            value=available_dbs[0] if available_dbs else "oxford5k",
            label="选择数据库",
            scale=1,
        )

        available_models = get_available_models()
        model_dropdown = gr.Dropdown(
            choices=available_models,
            value=available_models[0] if available_models else "oxford5k_200_AKM",
            label="选择模型",
            scale=1,
        )

        load_resources_button = gr.Button("加载选定资源", scale=1)
        refresh_button = gr.Button("刷新可用资源列表", scale=1)

    with gr.Row():
        status_textbox = gr.Textbox(
            value=initial_setup_message,
            label="System Status",
            interactive=False,
            scale=3,
        )
        rebuild_button = gr.Button("重建索引/模型", scale=1)

    # 查询区域
    with gr.Row():
        gr.Markdown("## 图像检索")

    with gr.Row():
        query_img_input = gr.Image(type="filepath", label="Query Image", scale=1)
        with gr.Column(scale=2):
            enable_spatial_cb = gr.Checkbox(
                label="Enable Spatial Re-ranking", value=True
            )
            gallery_output = gr.Gallery(
                label="Retrieved Images",
                show_label=True,
                elem_id="gallery",
                columns=[TOP_N_FINAL_RESULTS],
                rows=[1],
                object_fit="contain",
                height="auto",
            )
            scores_output = gr.Textbox(label="Scores / Info", lines=TOP_N_FINAL_RESULTS)

    # 事件绑定
    def refresh_resources():
        new_dbs = get_available_databases()
        new_models = get_available_models()
        return (
            gr.Dropdown(choices=new_dbs, value=new_dbs[0] if new_dbs else None),
            gr.Dropdown(
                choices=new_models, value=new_models[0] if new_models else None
            ),
            "资源列表已刷新",
        )

    refresh_button.click(
        refresh_resources, outputs=[db_dropdown, model_dropdown, status_textbox]
    )

    load_resources_button.click(
        setup_system_with_params,
        inputs=[db_dropdown, model_dropdown, gr.State(False)],
        outputs=[status_textbox],
    )

    query_img_input.upload(
        search_interface_with_params,
        inputs=[query_img_input, enable_spatial_cb, db_dropdown, model_dropdown],
        outputs=[gallery_output, scores_output],
    )

    query_img_input.change(
        lambda: (None, None), outputs=[gallery_output, scores_output]
    )

    rebuild_button.click(
        lambda db, model: setup_system_with_params(db, model, force_rebuild=True),
        inputs=[db_dropdown, model_dropdown],
        outputs=[status_textbox],
    )


if __name__ == "__main__":
    if vocabulary is None or tfidf_matrix_db is None or tfidf_transformer_db is None:
        print(
            "\nWARNING: Models not loaded. The system might not function correctly until models are built."
        )
        print(
            "Please add images to 'database_images' and consider 'Rebuild Index/Models'.\n"
        )
    demo.launch()
