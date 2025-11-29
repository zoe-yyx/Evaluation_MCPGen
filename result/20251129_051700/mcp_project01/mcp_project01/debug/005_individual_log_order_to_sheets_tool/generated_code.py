def log_order_to_sheets(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    import os
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    # Load environment variables
    SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
    SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

    # Authenticate and create a service for Google Sheets
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    service = build('sheets', 'v4', credentials=credentials)

    # Prepare the order data to be logged
    order_data = step.get("step_data", {}).get("step_8", [])
    if not order_data:
        return {"status": "error", "message": "No order data found"}

    # Define the range and values to be appended
    range_name = 'Orders!A1'  # Adjust the range as necessary
    values = [order_data]

    # Create the body for the request
    body = {
        'values': values
    }

    # Append the data to the Google Sheet
    try:
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()
        return {"status": "success", "message": "Order logged successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}