## Image Latent Representation

This project explores how a neural network can learn structure from images rather than just memorising pixels.

Instead of predicting labels, the model is trained to compress and reconstruct images.
Because it is forced to compress them into a small vector, it has to keep only the important visual information (texture, density, clustering).

## What the model learns

Each image is converted into a small numerical “fingerprint” (latent vector).
If the model learned meaningful structure, visually similar images should end up close together in this latent space.

To verify this, the embeddings are projected into 2D using PCA and plotted.

See: outputs/embedding_plot.png

## Why this matters

Many scientific problems (e.g. material or powder analysis) depend on structure rather than object identity.
This project mirrors that workflow:

image → learned representation → measurable comparison

So instead of manually defining features, the network discovers them automatically.

## Pipeline

Preprocess images into tensors

Train convolutional autoencoder (PyTorch)

Extract latent embeddings

Visualise clustering in 2D space

## How to run
pip install -r requirements.txt
python src/train_autoencoder.py

Then open:

notebook/02_embeddings.ipynb

and run all cells.

## Main takeaway

The clustering in the latent space shows the network captured structural similarity, not just pixel values.
This demonstrates how learned representations from images can be used as quantitative descriptors for real-world data.