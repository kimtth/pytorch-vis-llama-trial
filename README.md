## Attempts to configure Visual-LLaMA and Pytorch trials

- Work in progress:
Attempts to configure [Visual-LLaMA](https://github.com/feizc/Visual-LLaMA)
- Working directory: `vsllm`
- Created `dataset` and `model` directories, and downloaded data & models into them.
- Replaced from LLama v1 to v2

    ```cmd
    +---dataset
    |   |    
    |   +---coco2017
    |   |   +---annotations
    |   |   +---test2017
    |   |   +---train2017
    |   |   +---val2017
    |   |   +---test2017
    ```

    ```cmd
    +---model
    |   |    
    |   +---clip-vit-large-patch14
    |   |       ...
    |   |       tf_model.h5
    |   |       ...
    |   |       vocab.json
    |   |       
    |   +---Llama-2-7b-hf
    |   |       ...
    |   |       pytorch_model-00001-of-00002.bin
    |   |       pytorch_model-00002-of-00002.bin
    |   |       ...       
    ```

### Prerequisite

1. Install CUDA 12.1
1. Install Pytorch with CUDA support: `pip3 install torch --index-url https://download.pytorch.org/whl/cu121`
    - Ensure that CUDA version and Pytorch which support specific CUDA version should be matched.
    - Verify the result to confirm versions are matched: `torch.cuda.is_available()`.

### 1. Download Clip and Llama2

- Download [clip](https://huggingface.co/openai/clip-vit-large-patch14), 12.7 GB

    ```cmd
    git lfs clone https://huggingface.co/openai/clip-vit-large-patch14
    ```

    > Be aware to install Git extension for versioning large files. [url](https://git-lfs.com/), then initialize it by `git lfs install`.

- Download [llama2](https://huggingface.co/meta-llama/Llama-2-7b), 

1. Log in to HuggingFace. Once logged in, locate the Profile option positioned in the upper right corner and select Settings.
Image by the author

1. Choose the Access Tokens option and click the New token button in order to generate the token.

    ```cmd
    export GIT_LFS_SKIP_SMUDGE=1
    git lfs clone https://<USERNAME>:<YOUR_ACCESS_TOKEN>@huggingface.co/meta-llama/Llama-2-7b-hf
    ```

    In Windows, `set GIT_LFS_SKIP_SMUDGE=1` to resolve the issue of Filtering Content [url](https://stackoverflow.com/questions/53321680/what-does-filtering-content-mean-when-doing-a-git-clone).

    e.g., `git lfs clone https://hahaha:hf_nXXXXXXXXXXXXXXXX@huggingface.co/meta-llama/Llama-2-7b-hf`: The `hf` means HuggingFace format.

### 2.  Download dataset

- Visual-LLaMa supports `COCO` and `VQAV2` dataset. Entire size of `HuggingFaceM4/COCO` dataset is 25GB
- HuggingFaceM4/COCO dataset: [url](https://huggingface.co/datasets/HuggingFaceM4/COCO)

1. Download datasets in a directory

    ```cmd
    python -m pip install datasets
    ```

    ```python
    import os
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceM4/COCO")
    download_path = os.path.join(os.getcwd(), "dataset")
    dataset.save_to_disk(download_path)
    ```
    
    or

1. Download COCO dataset on Kaggle

    - Better speed than downloading by HuggingFace datasets library.

    - Kaggle [URL](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) for the COCO 2017 dataset

    - Extract downloaded zip file in the directory.

### 3.  Run the preprocessing script for train data

- Generate ImageEmbedding data as a pickle format for train data

    - Execute `preprocess.py`

### 4.  Run the SFT train script

- Execute `train.py`

### 5.  Run the script for fine-tuning

- Execute `finetune.py`

### 6.  Run and test for the inference

- Execute `infer.py`

### @TODO: 7. Code for setup Azure Machine Learning

## Reference

- Azure Multi-modal embedding API: [URL](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/image-retrieval)

- https://community.openai.com/t/text-embeddings-vs-word-embeddings/

    ![Alt text](docs/image.png)

- https://github.com/tonybeltramelli/pix2code

- https://github.com/balakreshnan/Samples2023/blob/main/LLM/llama2finetuning.md

- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-vs-code?view=azureml-api-2

- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2