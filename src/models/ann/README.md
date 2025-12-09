## Embedding Manager
Purpose

The Embedding Manager handles all logic related to storing and organizing item embeddings.
Its responsibilities include:

Receiving CLIP-generated visual embeddings from cropped clothing items

Routing embeddings to the appropriate index category (tops / bottoms / shoes)

Maintaining an item_id → vector_id mapping

Writing metadata + vector mappings into the database

Inputs

CLIP embedding for a single clothing item

Item metadata (category, color, pattern, etc.)

Outputs

Embedding inserted into the correct FAISS index

Persisted item_id ↔ vector_id mapping in the database

Upstream / Downstream

Upstream: receives embeddings from YOLO+CLIP inference

Downstream: communicates with the Index Builder to persist index updates

## Index Builder & Loader
Purpose

This module is responsible for constructing, storing, and loading FAISS indexes for all clothing categories.

Responsibilities include:

Building FAISS indexes (HNSW, IVF, or Flat)

Persisting indexes inside Docker volumes

Loading indexes into memory when the ANN container starts

Supporting index rebuilding and incremental updates

Inputs

Embeddings inserted by the Embedding Manager

Index file paths for each category

Outputs

Memory-loaded FAISS index objects for tops/bottoms/shoes

Upstream / Downstream

Upstream: receives vectors inserted by the Embedding Manager

Downstream: provides index objects to the ANN Search Service

## ANN Search Service
Purpose

The ANN Search Service performs approximate nearest neighbor search in FAISS.
Given an anchor item embedding, it retrieves compatible candidates from each category-specific FAISS index.

Responsibilities include:

Executing ANN search for bottoms and shoes given a top anchor

Returning top-K candidates with distance scores

Converting vector_id → item_id using the stored mappings

Inputs

Anchor item embedding (CLIP vector)

Category to search (e.g., "bottoms" or "shoes")

FAISS index objects

Outputs

Example output structure:

{
  "category": "bottoms",
  "results": [
    {"item_id": "bottom_023", "distance": 0.18},
    {"item_id": "bottom_041", "distance": 0.29}
  ]
}

Upstream / Downstream

Upstream: uses indexes loaded by Index Loader

Downstream: provides results to the Post-Search Reranking module and the Combination Generator

## Post-Search Filter & Reranking
Purpose

After ANN search returns candidates, this module applies semantic and metadata-based refinement:

Filters based on metadata (color, material, pattern, formality hints)

Incorporates BERT-based text intent (e.g., “formal”, “streetwear”, “date-night”)

Reranks candidate items based on a combined score

Inputs

ANN candidate lists from the ANN Search Service

User text intent classification (BERT output)

Clothing metadata (from the database)

Outputs

Filtered and reranked lists of bottoms and shoes

Clean candidate sets ready for the Outfit Combination Generator

Upstream / Downstream

Upstream: receives raw ANN candidates

Downstream: feeds optimized candidates into the outfit generation module

## How the Four Modules Work Together

Below is the end-to-end pipeline showing how all modules cooperate:

YOLO → item crops
     ↓
CLIP → visual embeddings
     ↓
[1] Embedding Manager
     - category routing
     - store item_id ↔ vector_id
     - insert vector into FAISS index
     ↓
[2] Index Builder & Loader
     - build and load FAISS indexes (tops/bottoms/shoes)
     ↓
[3] ANN Search Service
     - ANN(top embedding → bottoms.index)
     - ANN(top embedding → shoes.index)
     - returns top-K compatible items
     ↓
[4] Post-Search Filter & Reranking
     - metadata filtering
     - text-intent reranking (BERT)
     ↓
Outfit Combination Generator
     ↓
Final Outfit Recommendations (UI)

## System-Level Inputs & Outputs
System-Level Inputs (from upper-layer pipeline)

YOLO-cropped clothing images

CLIP embeddings for each item

Item metadata from the database

User text query / intent (BERT-derived)

System-Level Outputs (to lower-layer modules/UI)

Compatible bottoms and shoes for each anchor item

Reranked candidate lists

Outfit-ready components used by the Outfit Generator

Final ranked outfits shown to the user

## running command
