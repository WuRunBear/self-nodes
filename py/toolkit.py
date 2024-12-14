import comfy.model_management

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

class SelfNodes_StopCurrentQueue(object):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (any_type, {}),
                "boolean": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (any_type,)

    FUNCTION = "stop"

    CATEGORY = "SelfNodes"

    def stop(self, any, boolean):
        if boolean:
          comfy.model_management.interrupt_current_processing(True)
        return (any,)

#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py

NODE_CLASS_MAPPINGS = {
    "停止当前队列": SelfNodes_StopCurrentQueue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "停止当前队列": "停止当前队列"
}