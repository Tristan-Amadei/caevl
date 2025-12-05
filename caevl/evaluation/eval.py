import sys
import faiss
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time
import os

from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from caevl.evaluation import models
from caevl.evaluation import parser
from caevl.evaluation.test_dataset import TestDataset


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    time_parts = [f"{days}d" if days > 0 else "",
                  f"{hours}h" if hours > 0 else "",
                  f"{minutes}m" if minutes > 0 else "",
                  f"{seconds:.02f}s" if seconds > 0 else ""]
    return ' '.join(part for part in time_parts if part)


def make_dir(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


def main(args):

    logger.remove()  # Remove possibly previously existing loggers
    if not os.path.exists(Path("results") / args.log_dir):
        nb_folder_log = "0"
    else:
        nb_folder_log = str(len(os.listdir(Path("results") / args.log_dir)))
    log_dir = Path("results") / args.log_dir / nb_folder_log
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.get_model(args.method, args.backbone, args.descriptors_dimension,
                             args.device, weights=args.weights)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        args.database_coords_path,
        args.queries_coords_path,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        is_caevl='caevl' in args.method,
    )
    logger.info(f"Testing on {test_ds}")

    start = time.time()

    with torch.inference_mode():
        logger.debug("Extracting reference descriptors.")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        try:
            all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        except:
            all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float16")

        for images, indices, _ in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors.")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)
        for images, indices, _ in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database:]
    database_descriptors = all_descriptors[: test_ds.num_database]

    use_cosine_similarity = test_ds.is_caevl  # the CAEVl method uses the cosine similarity as score, other methods use L2
    if use_cosine_similarity:
        queries_descriptors = normalize(queries_descriptors)
        database_descriptors = normalize(database_descriptors)

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    if use_cosine_similarity:
        faiss_index = faiss.IndexFlatIP(args.descriptors_dimension)
    else:
        faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)

    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")

    search_batch_size = (test_ds.num_queries // 100) + 1
    all_predictions = []

    # Wrap the range object with tqdm for a progress bar
    for i in tqdm(range(0, queries_descriptors.shape[0], search_batch_size), desc="Searching queries"):
        queries_batch = queries_descriptors[i: i + search_batch_size]
        _, predictions = faiss_index.search(queries_batch, max(args.recall_values))
        all_predictions.append(predictions)

    predictions = np.concatenate(all_predictions, axis=0)

    stop = time.time()
    time_spent = format_time(stop - start)
    logger.info(f'It took {time_spent} to run.\n')

    if args.save_predictions:
        np.save(log_dir / "predictions.npy", predictions)
        predicted_coords = np.zeros((predictions.shape[0], predictions.shape[1], 2))
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                index_prediction = predictions[i, j]
                utm = test_ds.database_utms[index_prediction]
                predicted_coords[i, j, 0] = utm[0]
                predicted_coords[i, j, 1] = utm[1]

        queries_coords = np.zeros((predictions.shape[0], 2))
        for i in range(predictions.shape[0]):
            utm = test_ds.queries_utms[i]
            queries_coords[i, 0] = utm[0]
            queries_coords[i, 1] = utm[1]

        predicted_distances = np.linalg.norm(queries_coords[:, np.newaxis, :] - predicted_coords, axis=-1)
        np.save(log_dir / "predictions_distances.npy", predicted_distances)

    # For each query, check if the predictions are correct
    positives_per_query = test_ds.get_positives()
    recalls = np.zeros((len(positives_per_query), len(args.recall_values)))
    for j, positives_per_query__distance in enumerate(positives_per_query):
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.isin(preds[:n], positives_per_query__distance[query_index])):
                    recalls[j, i:] += 1
                    break

    # Divide by num_queries and multiply by 100, so the recalls are in percentages
    recalls = recalls / test_ds.num_queries * 100
    # recalls_str = ''
    # for j, positives_per_query__distance in enumerate(test_ds.positive_dist_threshold):
    #     recalls_str += f'\nRecalls@{positives_per_query__distance}:'
    #     for i, value in enumerate(args.recall_values):
    #         recalls_str += f' R@{value}: {recalls[j, i]:.2f}'

    # logger.info(recalls_str)

    def compute_top_k_accuracy_n_meters(query_coords, db_coords, predictions, k_values, n_values):
        """
        Computes top-k accuracy@n meters for multiple k values.

        Parameters:
            query_coords (numpy.ndarray): Query coordinates (num_queries, 2) [UTM East, UTM North]
            db_coords (numpy.ndarray): Database coordinates (num_db, 2) [UTM East, UTM North]
            k_values (list): List of k values to evaluate (e.g., [1, 5, 10])
            n (float): Distance threshold in meters

        Returns:
            dict: Top-k accuracy@n meters for each k in k_values.
        """
        # Extract the coordinates of the retrieved top-k database images
        retrieved_coords = db_coords[predictions]  # Shape: (num_queries, max_k, 2)

        # Compute Euclidean distances between query and retrieved coordinates
        query_coords_expanded = query_coords[:, np.newaxis, :]  # (num_queries, 1, 2)
        distances = np.linalg.norm(retrieved_coords - query_coords_expanded, axis=-1)  # (num_queries, max_k)

        # Compute accuracy for different k values
        accuracies = {}
        for n in n_values: accuracies[n] = {}
        for k in k_values:
            for n in n_values:
                # Check if at least one of the top-k predictions is within n meters
                correct_predictions = np.any(distances[:, :k] <= n, axis=1)
                accuracies[n][k] = np.mean(correct_predictions)

        return accuracies

    accuracies = compute_top_k_accuracy_n_meters(query_coords=test_ds.queries_utms,
                                                 db_coords=test_ds.database_utms,
                                                 predictions=predictions,
                                                 k_values=args.recall_values,
                                                 n_values=args.positive_dist_threshold)

    # format accuracies as a string
    accuracies_str = '\n'
    for n in args.positive_dist_threshold:
        accuracies_str += f'Acc@{n}m:\n'
        for k in args.recall_values:
            accuracies_str += f'   Top-{k}@{n}: {100*accuracies[n][k]:.2f}%\n'
        accuracies_str += '\n'

    logger.info(accuracies_str)


if __name__ == "__main__":
    args = parser.parse_arguments()

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main(args)
