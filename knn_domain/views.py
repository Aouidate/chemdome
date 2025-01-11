from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
from sklearn.neighbors import NearestNeighbors

matplotlib.use('agg')

def knn_domain(request):
    # Initialize variables
    train_points_in_ad = None
    test_points_in_ad = None
    Llabels_out_train = []
    Llabels_out_test = []
    plot_data = None
    error_message = None

    try:
        if request.method == 'POST' and 'training_file' in request.FILES and 'test_file' in request.FILES:
            # Read uploaded CSV files
            X_train = pd.read_csv(request.FILES['training_file'], index_col=0)
            X_test = pd.read_csv(request.FILES['test_file'], index_col=0)

            # Validate data
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Training and test datasets must have the same number of descriptors.")
            
            X_train.dropna(inplace=True)
            X_test.dropna(inplace=True)

            # Extract descriptors
            descriptors_train = X_train.values
            descriptors_test = X_test.values

            # Train k-NN model
            k = int(request.POST.get('k', 5))  # Default to 5
            nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nn_model.fit(descriptors_train)

            # Calculate distances
            distances_train, _ = nn_model.kneighbors(descriptors_train)
            distances_test, _ = nn_model.kneighbors(descriptors_test)

            avg_distance_train = np.mean(distances_train, axis=1)
            avg_distance_test = np.mean(distances_test, axis=1)

            # Define AD threshold
            threshold = np.percentile(avg_distance_train, 95)

            # Identify points in/out of AD
            train_in_ad = avg_distance_train <= threshold
            test_in_ad = avg_distance_test <= threshold

            Llabels_out_train = [label for label, in_ad in zip(X_train.index, train_in_ad) if not in_ad]
            Llabels_out_test = [label for label, in_ad in zip(X_test.index, test_in_ad) if not in_ad]

            # Generate plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(avg_distance_train, [0] * len(avg_distance_train), c='blue', label='Training Set', alpha=0.7)
            ax.scatter(avg_distance_test, [1] * len(avg_distance_test), c='red', label='Test Set', alpha=0.7)
            ax.axvline(x=threshold, color='k', linestyle='--', label=f'Threshold ({threshold:.2f})')
            ax.set_xlabel('Average Distance to k Nearest Neighbors')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Training', 'Test'])
            ax.legend()

            # Convert plot to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            plt.close(fig)

            # Calculate percentages
            train_points_in_ad = 100 * np.sum(train_in_ad) / len(train_in_ad)
            test_points_in_ad = 100 * np.sum(test_in_ad) / len(test_in_ad)

    except Exception as e:
        error_message = str(e)

    return render(request, 'knn_domain.html', {
        'train_points_in_ad': train_points_in_ad,
        'test_points_in_ad': test_points_in_ad,
        'Llabels_out_train': Llabels_out_train,
        'Llabels_out_test': Llabels_out_test,
        'plot_data': plot_data,
        'error_message': error_message
    })
