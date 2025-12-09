import json
from src.models.ann.index_builder_loader import IndexBuilderLoader

META_PATH = "data/user_embeddings/metadata.json"

def rebuild():
    print("🔄 Rebuilding FAISS indexes from metadata.json...")

    # 1) Load metadata.json
    with open(META_PATH, "r") as f:
        metadata_list = json.load(f)

    metadata_dict = {item["item_id"]: item for item in metadata_list}

    # 2) Initialize builder
    builder = IndexBuilderLoader(
        index_paths={
            "tops": "src/models/ann/faiss/tops.index",
            "bottoms": "src/models/ann/faiss/bottoms.index",
            "shoes": "src/models/ann/faiss/shoes.index",
        },
        dim=512,
        index_factory_string="Flat"
    )

    # 3) Ingest embeddings → add vectors into FAISS
    builder.ingest_items_from_metadata(metadata_dict)

    # 4) Save indexes & id_maps
    builder.save_all()

    print("🎉 FAISS rebuild complete!")


if __name__ == "__main__":
    rebuild()
