# utilities that may be helpful in completing the assignment. 

def save_and_share(model, filename='vae_model.safetensors', overwrite=True):
    """Save model to Google Drive and return shareable link"""
    from safetensors.torch import save_file
    from google.colab import drive, auth
    from googleapiclient.discovery import build
    import os
    
    # Mount drive if not already mounted
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    # Save model
    output_dir = '/content/drive/MyDrive/vae_models'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    if overwrite and os.path.exists(output_path): # overwrite: delete old version before saving
        os.remove(output_path)
        print(f"Deleted old version of {filename}")
    save_file(model.state_dict(), output_path)
    print(f"Model saved to: {output_path}")
    
    # Get shareable link
    auth.authenticate_user()
    service = build('drive', 'v3')
    
    results = service.files().list(
        q=f"name='{filename}'",
        fields='files(id)'
    ).execute()
    
    if results.get('files'):
        file_id = results['files'][0]['id']
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"Shareable link: {link}")
        return link
    
    return None
