"""
Train the NN model for specific locations or all locations (multi-location training supported).
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional
from process import process_scats_data
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential, Model
from keras.models import Model as KerasModel
from model.model import get_cnn, get_gru, get_lstm, get_saes
import warnings
warnings.filterwarnings("ignore")


def load_all_locations_data(identifiers: list[str], 
                            lags: int = 12, 
                            train_ratio: float = 0.8,
                            seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, MinMaxScaler]]:
    """Load and process data for all locations at once.
    
    # Arguments
        identifiers: List of location identifiers.
        lags: Integer, number of lag timesteps.
        train_ratio: Float, ratio of training data.
        seed: Optional integer, random seed.
    # Returns
        X_train, y_train, X_test, y_test, scalers_dict
    """
    vcols = [f"V{str(i).zfill(2)}" for i in range(96)]
    
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    scalers = {}
    
    print(f"Loading data for {len(identifiers)} locations...")
    
    for idx, identifier in enumerate(identifiers):
        try:
            df_id = pd.read_csv(f"data/scats/{identifier}.csv")
            flow = df_id[vcols].to_numpy(dtype=float).flatten()
            flow = np.array(np.nan_to_num(flow))
            
            cut = int(len(flow) * train_ratio)
            flow_train_raw = flow[:cut]
            flow_test_raw = flow[cut:]
            
            # Individual scaler per location
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_train_raw.reshape(-1, 1))
            scalers[identifier] = scaler
            
            flow_train = scaler.transform(flow_train_raw.reshape(-1, 1)).reshape(-1)
            flow_test = scaler.transform(flow_test_raw.reshape(-1, 1)).reshape(-1)
            
            # Create sequences
            for i in range(lags, len(flow_train)):
                all_X_train.append(flow_train[i - lags: i])
                all_y_train.append(flow_train[i])
            
            for i in range(lags, len(flow_test)):
                all_X_test.append(flow_test[i - lags: i])
                all_y_test.append(flow_test[i])
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(identifiers)} locations")
                
        except Exception as e:
            print(f"  Warning: Could not load {identifier}: {e}")
            continue
    
    X_train = np.array(all_X_train)
    y_train = np.array(all_y_train)
    X_test = np.array(all_X_test)
    y_test = np.array(all_y_test)
    
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    print(f"✓ Loaded {len(identifiers)} locations")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, scalers


def xiaochus_pipeline(identifier: str,
                      lags: int = 12,
                      train_ratio: float = 0.8,
                      seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Process data for a specific location identifier.
    
    # Arguments
        identifier: String, location identifier.
        lags: Integer, number of lag timesteps.
        train_ratio: Float, ratio of training data.
        seed: Optional integer, random seed.
    # Returns
        X_train, y_train, X_test, y_test, scaler
    """
    vcols = [f"V{str(i).zfill(2)}" for i in range(96)]
    df_id = pd.read_csv(f"data/scats/{identifier}.csv")
    flow = df_id[vcols].to_numpy(dtype=float).flatten()
    flow = np.array(np.nan_to_num(flow))
    cut = int(len(flow) * train_ratio)
    flow_train_raw = flow[:cut]
    flow_test_raw  = flow[cut:]
    
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_train_raw.reshape(-1, 1))

    flow_train = scaler.transform(flow_train_raw.reshape(-1, 1)).reshape(1, -1)[0]
    flow_test  = scaler.transform(flow_test_raw.reshape(-1, 1)).reshape(1, -1)[0]
    
    train_list, test_list = [], []
    for i in range(lags, len(flow_train)):
        train_list.append(flow_train[i - lags: i + 1])
    for i in range(lags, len(flow_test)):
        test_list.append(flow_test[i - lags: i + 1])
        
    train = np.array(train_list)
    test = np.array(test_list)
        
    if seed is not None:
        np.random.default_rng(seed)
        np.random.shuffle(train)
        
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    return X_train, y_train, X_test, y_test, scaler


def train_model_unified(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, 
                       name: str, config: dict[str, Any], save_path: str = "model/unified"):
    """Train a single unified model for all locations.
    
    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
        save_path: String, directory to save model.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    
    print(f"\nTraining unified {name.upper()} model...")
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1)

    # Create directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(f"{save_path}/{name}.keras")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"{save_path}/{name}_loss.csv", encoding='utf-8', index=False)
    
    print(f"✓ Unified model saved to {save_path}/{name}.keras")


def train_saes_unified(models: list[Sequential], X_train: np.ndarray, y_train: np.ndarray, 
                      name: str, config: dict[str, Any], save_path: str = "model/unified") -> None:
    """Train the SAEs model for all locations.
    
    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
        save_path: String, directory to save model.
    """
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = KerasModel(inputs=p.inputs,
                                           outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        
        print(f"Training SAE layer {i+1}/{len(models)-1}...")
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              verbose=1)
        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    train_model_unified(saes, X_train, y_train, name, config, save_path)


def train_model_location(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, 
                        name: str, identifier: str, config: dict[str, Any]):
    """Train a single model for a location.
    
    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        identifier: String, location identifier.
        config: Dict, parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1)

    # Create directory for location if it doesn't exist
    save_dir = Path(f"model/{identifier}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(f"model/{identifier}/{name}.keras")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"model/{identifier}/{name}_loss.csv", encoding='utf-8', index=False)
    
    print(f"✓ Model saved to model/{identifier}/{name}.keras")


def train_saes_location(models: list[Sequential], X_train: np.ndarray, y_train: np.ndarray, 
                       name: str, identifier: str, config: dict[str, Any]) -> None:
    """Train the SAEs model for a location.
    
    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        identifier: String, location identifier.
        config: Dict, parameter for train.
    """
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = KerasModel(inputs=p.inputs,
                                           outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              verbose=1)
        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    train_model_location(saes, X_train, y_train, name, identifier, config)


def train_pipeline_single(identifier: str, model_name: str, lag: int, config: dict[str, Any]) -> None:
    """Train a model for a specific location.
    
    # Arguments
        identifier: String, location identifier.
        model_name: String, name of the model to train.
        lag: Integer, number of lag timesteps.
        config: Dict, training configuration.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} for location: {identifier}")
    print(f"{'='*70}")
    
    try:
        X_train, y_train, _, _, _ = xiaochus_pipeline(identifier, lag)
        
        non_saes_layers = [lag, 64, 64, 1]
        saes_layers = [lag, 400, 400, 400, 1]
        
        if model_name == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = get_saes(saes_layers)
            train_saes_location(m, X_train, y_train, model_name, identifier, config)
            
        elif model_name == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_lstm(non_saes_layers)
            train_model_location(m, X_train, y_train, model_name, identifier, config)
            
        elif model_name == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_gru(non_saes_layers)
            train_model_location(m, X_train, y_train, model_name, identifier, config)
            
        elif model_name == 'cnn':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_cnn(non_saes_layers)
            train_model_location(m, X_train, y_train, model_name, identifier, config)
        
        print(f"✓ Completed training {model_name.upper()} for {identifier}\n")
        
    except Exception as e:
        print(f"✗ Error training {model_name} for {identifier}: {e}\n")
        raise


def train_pipeline_unified(identifiers: list[str], model_name: str, lag: int, config: dict[str, Any]) -> None:
    """Train a unified model for all locations.
    
    # Arguments
        identifiers: List of location identifiers.
        model_name: String, name of the model to train.
        lag: Integer, number of lag timesteps.
        config: Dict, training configuration.
    """
    print(f"\n{'='*70}")
    print(f"Training unified {model_name.upper()} model for {len(identifiers)} locations")
    print(f"{'='*70}")
    
    try:
        # Load all data at once
        X_train, y_train, X_test, y_test, scalers = load_all_locations_data(identifiers, lag)
        
        non_saes_layers = [lag, 64, 64, 1]
        saes_layers = [lag, 400, 400, 400, 1]
        
        if model_name == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = get_saes(saes_layers)
            train_saes_unified(m, X_train, y_train, model_name, config)
            
        elif model_name == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_lstm(non_saes_layers)
            train_model_unified(m, X_train, y_train, model_name, config)
            
        elif model_name == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_gru(non_saes_layers)
            train_model_unified(m, X_train, y_train, model_name, config)
            
        elif model_name == 'cnn':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = get_cnn(non_saes_layers)
            train_model_unified(m, X_train, y_train, model_name, config)
        
        # Save scalers
        import pickle
        save_dir = Path("model/unified")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(f"model/unified/{model_name}_scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        print(f"✓ Saved scalers for {len(scalers)} locations")
        
        print(f"✓ Completed training unified {model_name.upper()}\n")
        
    except Exception as e:
        print(f"✗ Error training unified {model_name}: {e}\n")
        raise


def main(argv):
    parser = argparse.ArgumentParser(description='Train models for traffic flow prediction at specific locations')
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "gru", "saes", "cnn", "all"],
        help="Model to train: lstm, gru, saes, cnn, or all (default: lstm)")
    parser.add_argument(
        "--location",
        default=None,
        help="Location identifier to train (default: all locations). Use 'all' or omit for all locations.")
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Train a single unified model for all locations instead of individual models")
    parser.add_argument(
        "--lag",
        type=int,
        default=12,
        help="Number of lag timesteps (default: 12)")
    parser.add_argument(
        "--batch",
        type=int,
        default=256,
        help="Batch size (default: 256)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Number of epochs (default: 600)")
    parser.add_argument(
        "--list-locations",
        action="store_true",
        help="List all available location identifiers and exit")
    
    args = parser.parse_args()

    # Get all location identifiers
    df = process_scats_data("./data/Scats Data October 2006.xls")
    all_identifiers = sorted([str(identifier) for identifier in df["Identifier"].unique()])
    
    # List locations if requested
    if args.list_locations:
        print(f"Available locations ({len(all_identifiers)}):")
        for i, loc in enumerate(all_identifiers, 1):
            print(f"  {i:3d}. {loc}")
        return
    
    # Determine which locations to train
    if args.location is None or args.location.lower() == "all":
        identifiers = all_identifiers
        print(f"Training on all {len(identifiers)} locations")
    else:
        if args.location not in all_identifiers:
            print(f"Error: Location '{args.location}' not found.")
            print(f"Use --list-locations to see available locations.")
            sys.exit(1)
        identifiers = [args.location]
        print(f"Training on location: {args.location}")
    
    # Determine which models to train
    if args.model == "all":
        model_names = ["lstm", "gru", "cnn", "saes"]
        print(f"Training all models: {', '.join([m.upper() for m in model_names])}")
    else:
        model_names = [args.model]
        print(f"Training model: {args.model.upper()}")
    
    # Training configuration
    lag = args.lag
    config: dict[str, Any] = {"batch": args.batch, "epochs": args.epochs}
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'Unified (single model)' if args.unified else 'Individual (per location)'}")
    print(f"  Lag: {lag}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Epochs: {config['epochs']}")
    
    # UNIFIED TRAINING MODE
    if args.unified:
        print(f"\n🚀 Starting unified training mode...")
        failed_tasks = []
        
        for model_name in model_names:
            try:
                train_pipeline_unified(identifiers, model_name, lag, config)
            except Exception as e:
                error_msg = f"Unified {model_name.upper()}"
                failed_tasks.append(error_msg)
                print(f"✗ Failed: {error_msg}")
                continue
        
        # Summary
        print(f"\n{'='*70}")
        print(f"UNIFIED TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total models: {len(model_names)}")
        print(f"Successful: {len(model_names) - len(failed_tasks)}")
        print(f"Failed: {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\nFailed tasks:")
            for task in failed_tasks:
                print(f"  ✗ {task}")
        else:
            print(f"\n✓ All unified models trained successfully!")
        
        print(f"{'='*70}\n")
        
    # INDIVIDUAL TRAINING MODE
    else:
        total_tasks = len(identifiers) * len(model_names)
        current_task = 0
        failed_tasks = []
        
        for identifier in identifiers:
            for model_name in model_names:
                current_task += 1
                print(f"\n{'#'*70}")
                print(f"Progress: {current_task}/{total_tasks}")
                print(f"{'#'*70}")
                
                try:
                    train_pipeline_single(identifier, model_name, lag, config)
                except Exception as e:
                    error_msg = f"{model_name.upper()} for {identifier}"
                    failed_tasks.append(error_msg)
                    print(f"✗ Failed: {error_msg}")
                    continue
        
        # Summary
        print(f"\n{'='*70}")
        print(f"INDIVIDUAL TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful: {total_tasks - len(failed_tasks)}")
        print(f"Failed: {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\nFailed tasks:")
            for task in failed_tasks:
                print(f"  ✗ {task}")
        else:
            print(f"\n✓ All tasks completed successfully!")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv)