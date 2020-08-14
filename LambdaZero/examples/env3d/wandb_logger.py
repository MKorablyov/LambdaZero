from typing import Dict

from wandb.ray import WandbLogger


def recursively_remove_classes_and_functions_from_dictionary(input_dict: Dict) -> Dict:
    """
     Remove problematic elements from dictionary. Address all levels of the
     dictionary recursively.

     Args:
         input_dict (Dict):  dictionary of parameters

     Returns:
         output_dict (Dict): copy of original_dict where any item where the value is a class
                             is removed.
     """
    output_dict = {}

    for key, value in input_dict.items():
        if isinstance(value, dict):
            output_dict[key] = recursively_remove_classes_and_functions_from_dictionary(
                value
            )
        elif not isinstance(value, type) and not callable(value):
            output_dict[key] = value

    return output_dict


class LambdaZeroWandbLogger(WandbLogger):
    """
    We have to derive the WandbLogger class because it cannot handle
    classes as configuration parameters.

    The necessity of this kluge is a direct consequence of passing classes and
    functions in what should be a config dictionary with strings and numbers as values.
    """

    def on_result(self, result):
        """
        This is a wrapper of the base class method which strips the result dictionary
        of problematic "class" arguments, which wandb cannot deal with.
        """
        # remove classes and functions at all levels recursively
        clean_result = recursively_remove_classes_and_functions_from_dictionary(result)

        super().on_result(clean_result)
