from django.shortcuts import render

# Create your views here.
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.decomposition import PCA
#matplotlib.use('agg')

def calculate_domain(request):

    if request.method == 'POST' and 'training_file' in request.FILES and 'test_file' in request.FILES:
        X_train = pd.read_csv(request.FILES['training_file'],index_col= 0)
        X_train.dropna(inplace=True)
        X_test = pd.read_csv(request.FILES['test_file'],index_col= 0)
        X_test.dropna(inplace=True)

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        pca = PCA()
        #X_trainS = pca.fit_transform(X_train)
        #X_testS = pca.transform(X_test)

        # Reduce from 4 to 2 features with PCA
        pca = PCA(n_components=3)
 
        # Fit and transform data
        pca_features_train = pca.fit_transform(X_train)
        pca_features_test = pca.transform(X_test)

        # Create dataframe
        pca_df_train = pd.DataFrame(
        data=pca_features_train, 
        columns=['PC1', 'PC2', 'PC3'])
 
        pca_df_test = pd.DataFrame(
        data=pca_features_test, 
        columns=['PC1', 'PC2', 'PC3'])

        # Calculate the PCA bounding box limits
        pc1_max = np.max(pca_df_train.iloc[:, 0])
        pc1_min = np.min(pca_df_train.iloc[:, 0])
        pc2_max = np.max(pca_df_train.iloc[:, 1])
        pc2_min = np.min(pca_df_train.iloc[:, 1])

        # Plot each data point with the corresponding color
        fig, ax = plt.subplots(figsize=(10, 8))  # Set the figure size
        im = ax.scatter(pca_df_train["PC1"], pca_df_train["PC2"], label='Training Set', marker= 'D', c = 'b')
        im = ax.scatter(pca_df_test["PC1"], pca_df_test["PC2"], label='Test Set', c = "r", marker= r'$\clubsuit$', s= 150)

        # Add the PCA bounding box to the plot

        # Create a line between the four values
        line = np.array([[pc1_min, pc2_min], [pc1_max, pc2_min], [pc1_max, pc2_max], [pc1_min, pc2_max]])

        # Close the loop by adding the first point to the end of the line
        line = np.append(line, [[pc1_min, pc2_min]], axis=0)

        # Plot the line
        plt.plot(line[:, 0], line[:, 1], color='black', linewidth=1)

        #sns.scatterplot(data=pca_df, x='PC1', y='PC2', palette=color_mapping, sizes= 150)

        # Add labels to the data points
        #for i, txt in enumerate(pca_df['classes']):
        #    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=10)

        # Customize plot title and axes labels
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.legend(loc='lower center', bbox_to_anchor =(0.5,-0.15),ncol=2)
        plt.tight_layout()
        plt.show()


        # Convert the Matplotlib figure to a base64-encoded string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        plt.close(fig)  # Close the figure to release memory


        return render(request, 'result1.html', {'plot_data': plot_data })  # Pass the plot data to the template
    else:
        return render(request, 'upload1.html')
