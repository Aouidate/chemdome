from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64

matplotlib.use('agg')

def calculate_domain(request):
    # Initialize variables to hold the results, in case the form isn't submitted yet
    train_points_in_ad = None
    test_points_in_ad = None
    Llabels_out_train = []
    Llabels_out_test = []
    plot_data = None

    if request.method == 'POST' and 'training_file' in request.FILES and 'test_file' in request.FILES:
        # Read uploaded CSV files
        X_train = pd.read_csv(request.FILES['training_file'], index_col=0)
        X_train.dropna(inplace=True)
        X_test = pd.read_csv(request.FILES['test_file'], index_col=0)
        X_test.dropna(inplace=True)

        # Perform applicability domain calculation using the data
        num_descriptors = X_train.shape[1] - 2  # Number of descriptors in the model
        num_molecules = X_train.shape[0]  # Number of molecules in the training set

        ### TRAINING SET ###
        labels_train = [lbtr for lbtr in X_train.index]

        # Calculate the standardized residuals for the training set
        last_column_index_train = X_train.shape[1] - 1  # Index of the last column
        actual_values_train = X_train.iloc[:, -1]
        predicted_values_train = X_train.iloc[:, last_column_index_train - 1]
        residuals_train = np.array(actual_values_train) - np.array(predicted_values_train)
        std_residuals_train = residuals_train / np.std(residuals_train)

        # Calculate the leverage values for the training set
        X_train_inv = np.linalg.inv(np.dot(X_train.T, X_train))
        leverage_values_train = np.diag(np.dot(np.dot(X_train, X_train_inv), X_train.T))

        ### TEST SET ###
        labels_test = [lbts for lbts in X_test.index]

        # Calculate the standardized residuals for the test set
        last_column_index_test = X_test.shape[1] - 1  # Index of the last column
        actual_values_test = X_test.iloc[:, -1]
        predicted_values_test = X_test.iloc[:, last_column_index_test - 1]
        residuals_test = np.array(actual_values_test) - np.array(predicted_values_test)
        std_residuals_test = residuals_test / np.std(residuals_test)

        # Calculate the leverage values for the test set
        leverage_values_test = np.diag(np.dot(np.dot(X_test, X_train_inv), X_test.T))

        # Calculate the h* threshold based on the number of molecules in the training set
        threshold = 3 * (num_descriptors + 1) / num_molecules

        # Plot the filtered standardized residuals vs leverage values for the training set
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.scatter(leverage_values_test, std_residuals_test, label='Test Set', c="r", marker=r'$\clubsuit$', s=150)
        im = ax.scatter(leverage_values_train, std_residuals_train, label='Training Set', marker='D', c=actual_values_train, cmap=plt.cm.jet)

        # Add numbers to scatter points for the train set 
        Llabels_out_train = []
        for label, x, y in zip(labels_train, leverage_values_train, std_residuals_train):
            if (y > 3 or y < -3) or (x > threshold):
                Llabels_out_train.append(label)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Add numbers to scatter points for the test set 
        Llabels_out_test = []
        for label, x, y in zip(labels_test, leverage_values_test, std_residuals_test):
            if (y > 3 or y < -3) or (x > threshold):
                Llabels_out_test.append(label)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.axhline(y=3, color='r', linestyle='-')
        plt.axhline(y=-3, color='r', linestyle='-')
        plt.axvline(x=threshold, color='k', linestyle='--')
        plt.xlabel('Leverage (hi) h*='f'{threshold}')
        plt.ylabel('Standardized Residuals')
        plt.title('Williams Plot')
        fig.colorbar(im, ax=ax)
        plt.legend(loc='upper right')

        # Convert the Matplotlib figure to a base64-encoded string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        plt.close(fig)  # Close the figure to release memory

        # Calculate percentage of points in AD
        train_points_in_ad = float(100 * np.sum(np.asarray(leverage_values_train < threshold) & np.asarray(std_residuals_train < 3))) / len(leverage_values_train)
        test_points_in_ad = float(100 * np.sum(np.asarray(leverage_values_test < threshold) & np.asarray(std_residuals_test < 3))) / len(leverage_values_test)

    # Render the template with either the form or the results
    return render(request, 'leverage_domain.html', {
        'train_points_in_ad': train_points_in_ad,
        'test_points_in_ad': test_points_in_ad,
        'Llabels_out_train': Llabels_out_train,
        'Llabels_out_test': Llabels_out_test,
        'plot_data': plot_data
    })
