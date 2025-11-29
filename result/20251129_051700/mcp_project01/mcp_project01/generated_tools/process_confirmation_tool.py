def process_confirmation(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    user_input = context.get("conversation_context", {}).get("user_input", "").strip().lower()
    order_data = context.get("order_data")
    
    if user_input in ["yes", "confirm"]:
        confirmation_status = "confirmed"
        response_message = "Your order has been confirmed."
    elif user_input in ["no", "cancel"]:
        confirmation_status = "canceled"
        response_message = "Your order has been canceled."
    else:
        confirmation_status = "pending"
        response_message = "Please respond with 'yes' to confirm or 'no' to cancel your order."
    
    return {
        "confirmation_status": confirmation_status,
        "response_message": response_message,
        "order_data": order_data,
        "step_data": context.get("step_data", {})
    }