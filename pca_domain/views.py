from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.decomposition import PCA

def calculate_domain(request):
    plot_data = None
    train_points_in_ad = None
    test_points_in_ad = None
    Llabels_out_train = []
    Llabels_out_test = []

    if request.method == 'POST' and 'training_file' in request.FILES and 'test_file' in request.FILES:
        # Load training and test data from uploaded CSV files
        X_train = pd.read_csv(request.FILES['training_file'], index_col=0)
        X_train.dropna(inplace=True)
        X_test = pd.read_csv(request.FILES['test_file'], index_col=0)
        X_test.dropna(inplace=True)

        # Standardize the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # PCA to reduce from 4 to 2 components (or 3, as needed)
        pca = PCA(n_components=3)
        pca_features_train = pca.fit_transform(X_train)
        pca_features_test = pca.transform(X_test)

        # Create DataFrame for PCA features
        pca_df_train = pd.DataFrame(data=pca_features_train, columns=['PC1', 'PC2', 'PC3'])
        pca_df_test = pd.DataFrame(data=pca_features_test, columns=['PC1', 'PC2', 'PC3'])

        # Calculate PCA bounding box limits (based on training data)
        pc1_max = np.max(pca_df_train['PC1'])
        pc1_min = np.min(pca_df_train['PC1'])
        pc2_max = np.max(pca_df_train['PC2'])
        pc2_min = np.min(pca_df_train['PC2'])

        # Plotting PCA with bounding box
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pca_df_train['PC1'], pca_df_train['PC2'], label='Training Set', marker='D', c='b')
        ax.scatter(pca_df_test['PC1'], pca_df_test['PC2'], label='Test Set', c='r', marker=r'$\clubsuit$', s=150)

        # Add PCA bounding box to the plot
        bounding_box = np.array([[pc1_min, pc2_min], [pc1_max, pc2_min], [pc1_max, pc2_max], [pc1_min, pc2_max]])
        bounding_box = np.append(bounding_box, [[pc1_min, pc2_min]], axis=0)  # Close the loop
        plt.plot(bounding_box[:, 0], bounding_box[:, 1], color='black', linewidth=1)

        # Plotting customizations
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()

        # Save the figure as base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close(fig)

        # Calculate percentage of points inside AD (within the bounding box)
        # For simplicity, assuming points within the bounding box are inside the AD
        train_points_in_ad = np.sum((pca_df_train['PC1'] >= pc1_min) & (pca_df_train['PC1'] <= pc1_max) &
                                    (pca_df_train['PC2'] >= pc2_min) & (pca_df_train['PC2'] <= pc2_max)) / len(pca_df_train) * 100
        test_points_in_ad = np.sum((pca_df_test['PC1'] >= pc1_min) & (pca_df_test['PC1'] <= pc1_max) &
                                   (pca_df_test['PC2'] >= pc2_min) & (pca_df_test['PC2'] <= pc2_max)) / len(pca_df_test) * 100

        # Identify outliers (points outside the bounding box)
        Llabels_out_train = pca_df_train[~((pca_df_train['PC1'] >= pc1_min) & (pca_df_train['PC1'] <= pc1_max) &
                                           (pca_df_train['PC2'] >= pc2_min) & (pca_df_train['PC2'] <= pc2_max))]['PC1'].index.tolist()
        Llabels_out_test = pca_df_test[~((pca_df_test['PC1'] >= pc1_min) & (pca_df_test['PC1'] <= pc1_max) &
                                         (pca_df_test['PC2'] >= pc2_min) & (pca_df_test['PC2'] <= pc2_max))]['PC1'].index.tolist()

        # Pass all the results to the template
        context = {
            'plot_data': plot_data,
            'train_points_in_ad': train_points_in_ad,
            'test_points_in_ad': test_points_in_ad,
            'Llabels_out_train': Llabels_out_train,
            'Llabels_out_test': Llabels_out_test
        }

        return render(request, 'pca_domain.html', context)

    # If it's a GET request or no files were uploaded, render the form
    return render(request, 'pca_domain.html')
