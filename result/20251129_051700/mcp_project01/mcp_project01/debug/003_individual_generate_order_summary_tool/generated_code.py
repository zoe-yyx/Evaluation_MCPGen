def generate_order_summary(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    order_details = step.get('step_6', {})
    summary = {
        "order_id": order_details.get("order_id", "N/A"),
        "customer_name": order_details.get("customer_name", "N/A"),
        "items": order_details.get("items", []),
        "total_amount": order_details.get("total_amount", 0.0),
        "status": "Pending Confirmation"
    }
    return summary