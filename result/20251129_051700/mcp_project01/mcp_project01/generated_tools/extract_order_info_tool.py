async def extract_order_info(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    user_input = step.get("input", {}).get("user_input", "")
    order_details = {
        "items": [],
        "table_number": None
    }
    
    if "table" in user_input:
        table_info = user_input.split("table")
        order_details["table_number"] = table_info[1].strip() if len(table_info) > 1 else None
        user_input = table_info[0].strip()
    
    items = user_input.split("and")
    for item in items:
        item = item.strip()
        if " " in item:
            quantity, drink = item.split(" ", 1)
            order_details["items"].append({
                "quantity": int(quantity),
                "drink": drink.strip()
            })
    
    return order_details