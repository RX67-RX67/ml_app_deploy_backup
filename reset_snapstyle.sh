#!/bin/bash

echo "🔄 Resetting SnapStyle user data..."
BASE="./data/user_embeddings"
CROPS="./data/user_crops"
FAISS="./src/models/ann/faiss"

echo "--------------------------------------------"
echo "🗑️ 1. Removing old metadata.json..."
rm -f $BASE/metadata.json

echo "--------------------------------------------"
echo "🗑️ 2. Removing ALL embedding .npy files..."
find $BASE -type f -name "*.npy" -delete

echo "--------------------------------------------"
echo "🗑️ 3. Removing ALL YOLO crop images..."
find $CROPS -type f -name "*.jpg" -delete
find $CROPS -type f -name "*.png" -delete

echo "--------------------------------------------"
echo "🗑️ 4. Removing FAISS indexes..."
rm -f $FAISS/tops.index
rm -f $FAISS/bottoms.index
rm -f $FAISS/shoes.index
rm -f $FAISS/id_maps.json

echo "--------------------------------------------"
echo "📁 5. Recreating empty structure..."
mkdir -p $BASE
mkdir -p $CROPS
mkdir -p $FAISS

echo "--------------------------------------------"
echo "✨ Reset complete! Your SnapStyle backend is now clean."
echo "👉 Next step: run  'docker compose down && docker compose up --build'"
