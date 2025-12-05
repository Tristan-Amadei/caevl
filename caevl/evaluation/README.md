# Evaluation
This folder contains the evaluation code used to benchmark our model.
It is based on and adapted from the excellent work of the authors of the original repository:

🔗 Original source: [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation)

We gratefully acknowledge and thank the original authors for making their code available and open to modification.
Our version includes modifications, extensions, and refactoring tailored to our model and experiments.

## Run evaluation
Here is an example of how evaluate the model
```
python3 caevl/evaluation/eval.py \
    --method=caevl \
    --weights=caevl/models/trained/stagetwo/stagetwo.pth \
    --database_folder=path/to/database \
    --queries_folder=path/to/queries \
    --database_coords_path=path/to/database_coords \
    --queries_coords_path=path/to/queries_coords \
```

This will create a log file in logs/log_dir. You can add --save_predictions to save the predictions, allowing you to visualize and analyze them afterwards.<br>
The --database_coords_path and --queries_coords_path parameters are paths to the dictionaries that contain the coordinates of the database and query images. They need to have the names of images as keys and the coordinates as values. Otherwise, the coordinates of the images can be stored directly in the filenames, as such @utm