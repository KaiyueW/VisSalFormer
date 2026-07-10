import open_clip # CLIP vision-language model (image encoder here)
import torch
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer # text embedding model
import logging

logger = logging.getLogger(__name__)

class ChartQADataset(Dataset):
    def __init__(self, chart_dir, saliency_dir, json_path):
        self.chart_dir = chart_dir
        self.saliency_dir = saliency_dir

        with open(json_path, 'r') as f:
            self.samples = json.load(f)
            # get {imgname, query, label, saliency_map}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chart_path = os.path.join(self.chart_dir, sample['imgname']) # get full img path
        chart = Image.open(chart_path).convert('RGB')
        saliency_path = os.path.join(self.saliency_dir, sample['saliency_map'])
        saliency = Image.open(saliency_path).convert('RGB')
        return {
            "image":   chart,
            "imgname": sample["imgname"],
            "saliency_img": saliency,
            "saliency_name": sample["saliency_map"],
            "query":   sample["query"],
            "label":   sample["label"],
        }

def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch
    # return {"image": [img1, img2, ...], "imgname": [...], ...}


class ChartQASelector:
    def __init__(
        self,
        train_dataset: ChartQADataset,
        device: str = "cuda",
        batch_size: int = 32,
        dinov2_model: str = "facebook/dinov2-large",
        text_model: str = "sentence-transformers/sentence-t5-base" / all-mpnet-base-v2
        cached_features=None,
        similar_in_topk: int = 200,
    ):
        self.train_dataset = train_dataset
        self.device = device
        self.batch_size = batch_size
        self.similar_in_topk = similar_in_topk

        # Load the model and processor
        logger.info(f"-----Loading the vision model: {dinov2_model}")
        self.image_processor = AutoImageProcessor.from_pretrained(dinov2_model)
        self.image_model = AutoModel.from_pretrained(dinov2_model).to(self.device)

        logger.info(f"-----Loading the text model: {text_model}")
        self.text_model = SentenceTransformer(text_model).to(self.device)

        # Precompute features
        if cached_features is None:
            self.train_image_features = self._precompute_features() # compute image embeddings
        else:
            logger.info("Using cached image features.")
            self.train_image_features = cached_features # load saved embeddings


    def _precompute_features(self, save_path: str):
        # Encode all train images with DINOv2 and cache the CLS token embeddings.
        logger.info("Precomputing the training datset image features.")
        train_image_features = []

        # Switch to evaluation mode
        self.image_model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features using DinoV2",
            ):
                inputs = self.image_processor(
                    images=batch["image"],
                    return_tensors="pt",
                ).to(self.device)
 
                outputs = self.image_model(**inputs) #each image → vector embedding
                # CLS token = index 0 of last_hidden_state
                cls_features = outputs.last_hidden_state[:, 0, :]
                cls_features /= cls_features.norm(dim=-1, keepdim=True) #normalize
                train_image_features.append(cls_features.detach())

        train_image_features = torch.cat(train_image_features) #concate
        train_image_features = train_image_features.to("cpu")

        torch.save(train_image_features, save_path)
        logger.info(f"Saved training image features into {save_path}.")
        return train_image_features


    def find_by_ranking_similar_text(self, batch_image, batch_text, num_examples, with_answers=False, do_reverse=False): # dont include answers, just question
        """
        RICES Images -> rank based on text similarity

        Args:
            batch ():
            num_examples ():

        Returns:

        """
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch_image]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)


            # logger.debug(f"query_feature shape: {query_feature.shape}")
            # logger.debug(f"self.features shape: {self.features.shape}")
            # Compute the similarity of the input image to the precomputed features (training dataset)
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)
            # logger.debug(f"similarity shape: {similarity.shape}")
            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :self.similar_in_topk]
            # all queries, and only first top-k indices
            rices_samples = [[self.dataset[i] for i in reversed(row)] for row in indices]
            # indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]
            # return [[self.dataset[i] for i in reversed(row)] for row in indices]

            # rank based on text similarity

            text_inputs = [text for text in batch_text]
            # logger.debug(f"text_inputs: {text_inputs}")
            text_query_features = self.text_model.encode(
                text_inputs,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            text_query_features /= text_query_features.norm(dim=-1, keepdim=True)

            if self.dataset.dataset_name in ["vqav2", "ok_vqa", "vizwiz", "gqa", "textvqa"]:
                if with_answers:
                    rices_samples_text = [[sample["question"] + " " + ", ".join(sample["answers"]) for sample in samples] for samples in rices_samples]
                else:
                    rices_samples_text = [[sample["question"] for sample in samples] for samples in rices_samples]
            elif self.dataset.dataset_name in ["coco", "flickr"]:
                rices_samples_text = [[sample["caption"] for sample in samples] for samples in rices_samples]
            else:
                raise NotImplementedError(f"dataset_name: {self.dataset.dataset_name} not supported")

            rices_samples_text_features = torch.stack([self.text_model.encode(
                sample_text,
                convert_to_tensor=True,
                show_progress_bar=False,
            ) for sample_text in rices_samples_text])

            rices_samples_text_features /= rices_samples_text_features.norm(dim=-1, keepdim=True)
            text_query_features = text_query_features.unsqueeze(dim=1)
            # logger.debug(f"rices_samples_text_features.shape: {rices_samples_text_features.shape}"
            #              f"text_query_features.shape: {text_query_features.shape}")
            # text_similarity = (text_query_features @ rices_samples_text_features.T).squeeze()
            text_similarity = torch.einsum("bij,bkj->bki", text_query_features, rices_samples_text_features)
            text_similarity = text_similarity.squeeze(dim=-1)
            # logger.debug(f"text_similarity.shape: {text_similarity.shape}")
            indices = text_similarity.argsort(dim=-1, descending=True)[:, :num_examples] # TODO,

            # demos = [[rices_samples[j][i] for i in reversed(row)] for j,row in enumerate(indices)]
            # sub_ind = torch.randperm(indices.shape[1])
            # indices = indices[:, sub_ind[:num_examples]]
            # logger.debug(f"indices.shape: {indices.shape}")
            # assert False
        if do_reverse:
            return [[rices_samples[j][i] for i in reversed(row)] for j,row in enumerate(indices)]
        else:
            return [[rices_samples[j][i] for i in row] for j,row in enumerate(indices)]


