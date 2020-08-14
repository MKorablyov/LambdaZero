from typing import Dict

from wandb.ray import WandbLogger


def remove_classes_and_functions_from_dictionary(original_dict: Dict) -> Dict:
    """
    Remove problematic elements from dictionary.

    Args:
        original_dict (Dict):  dictionary of parameters

    Returns:
        clean_dict (Dict): copy of original_dict where any item where the value is a class
                            is removed.
    """
    clean_dict = {}
    for key, value in original_dict.items():
        # check that value is neither a class or a function
        if not isinstance(value, type) and not callable(value):
            clean_dict[key] = value
    return clean_dict


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

        # remove classes and functions at first level
        clean_result = remove_classes_and_functions_from_dictionary(result)

        # remove classes and functions at second nested level
        for key, value in clean_result.items():
            if isinstance(value, dict):
                clean_result[key] = remove_classes_and_functions_from_dictionary(value)

                # remove classes and functions at third nested level
                for key2, value2 in value.items():
                    if isinstance(value2, dict):
                            clean_result[key][key2] = remove_classes_and_functions_from_dictionary(value2)

        super().on_result(clean_result)

