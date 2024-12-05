from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from io import BytesIO
import base64

def calculate_sali(request):
    # Initialize variables to hold the results
    sali_results = None
    plot_data = None
    error_message = None

    if request.method == 'POST' and 'file' in request.FILES:
        try:
            # Read the uploaded CSV file
            data = pd.read_csv(request.FILES['file'])
            data.columns = data.columns.str.lower()  # Normalize column names

            # Ensure required columns are present
            if 'smiles' not in data.columns or 'pic50' not in data.columns:
                error_message = "Uploaded file must contain 'SMILES' and 'pIC50' columns (case-insensitive)."
            else:
                # Generate fingerprints and validate molecules
                data['mol'] = [Chem.MolFromSmiles(x) for x in data.smiles]
                data['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in data.mol]
                if data['mol'].isnull().any():
                    error_message = "Invalid SMILES found in the dataset."
                elif not np.issubdtype(data['pic50'].dtype, np.number):
                    error_message = "'pIC50' column must contain numerical values."
                else:
                    # Retrieve user parameters
                    similarity_threshold = float(request.POST.get('similarity_threshold', 0.9))
                    activity_difference = float(request.POST.get('activity_difference', 2.0))

                    # Pairwise similarity calculations using RDKit
                    fps = data['fp'].tolist()
                    sali_data = []
                    for i, fp_i in enumerate(fps):
                        similarities = DataStructs.BulkTanimotoSimilarity(fp_i, fps[i + 1:])
                        for j, similarity in enumerate(similarities, start=i + 1):
                            delta = abs(data['pic50'].iloc[i] - data['pic50'].iloc[j])
                            if similarity > 0 and delta > 0:
                                sali = delta / similarity
                                sali_data.append({
                                    'Similarity': similarity,
                                    'Delta': delta,
                                    'SMILES_i': data['smiles'].iloc[i],
                                    'SMILES_j': data['smiles'].iloc[j],
                                    'pIC50_i': data['pic50'].iloc[i],
                                    'pIC50_j': data['pic50'].iloc[j],
                                    'SALI': sali
                                })

                    # Convert results to DataFrame and sort
                    sali_results = pd.DataFrame(sali_data).sort_values(by='SALI', ascending=False).head(10)

                    # Generate scatter plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(
                        sali_data['Similarity'], sali_data['Delta'],
                        c=sali_data['SALI'], cmap=plt.cm.viridis, #edgecolor='k'
                    )
                    plt.colorbar(scatter, ax=ax, label='SALI')
                    plt.axhline(y=activity_difference, color='red', linestyle='--', label="Activity Difference")
                    plt.axvline(x=similarity_threshold, color='blue', linestyle='--', label="Similarity Threshold")
                    plt.legend()
                    plt.xlabel('Similarity')
                    plt.ylabel('Activity Difference (Δ)')
                    plt.title('Activity Cliffs (SALI Plot)')

                    # Encode plot to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
                    buffer.close()
                    plt.close(fig)

        except Exception as e:
            error_message = f"Error in processing the file: {str(e)}"

    return render(request, 'sali.html', {
        'clifs': sali_results.to_dict('records') if sali_results is not None else None,
        'img_url': f"data:image/png;base64,{plot_data}" if plot_data else None,
        'error': error_message
    })
