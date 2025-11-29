async def generate_success_response(step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    response = {
        "status": "success",
        "message": "Your order has been successfully processed.",
        "conversation_state": context.get("conversation_state", "unknown")
    }
    return response