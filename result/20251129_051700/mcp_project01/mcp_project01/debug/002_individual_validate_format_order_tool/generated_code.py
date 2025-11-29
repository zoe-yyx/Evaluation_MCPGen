def validate_format_order(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    order_data = step.get("step_data", {}).get("step_4", {})
    
    if not isinstance(order_data, dict):
        return {"status": "error", "message": "Invalid order format. Expected a dictionary."}
    
    required_fields = ["restaurant_id", "items", "total_amount"]
    for field in required_fields:
        if field not in order_data:
            return {"status": "error", "message": f"Missing required field: {field}"}
    
    if not isinstance(order_data["items"], list) or not order_data["items"]:
        return {"status": "error", "message": "Items should be a non-empty list."}
    
    for item in order_data["items"]:
        if not isinstance(item, dict) or "item_id" not in item or "quantity" not in item:
            return {"status": "error", "message": "Each item must be a dictionary with item_id and quantity."}
    
    if not isinstance(order_data["total_amount"], (int, float)) or order_data["total_amount"] < 0:
        return {"status": "error", "message": "Total amount must be a non-negative number."}
    
    return {"status": "success", "message": "Order format is valid."}