from enum import Enum

class GraphUrls(Enum):
    token_url = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    token_scope = "https://graph.microsoft.com/.default"
    site_id_url = "https://graph.microsoft.com/v1.0/sites/{sp_site_name}:/sites/{site_name}"
    drive_id_url = "https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    files_by_path_url = "https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
    files_by_id_url = "https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
    file_content_url = "https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{file_id}/content"
    file_content_pdf_url = "https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{file_id}/content?format=pdf"