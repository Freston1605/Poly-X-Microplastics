"""Dataset loading and validation.

This package is the future home of the dataset module, which will own
``data.yaml`` class names and pass them to the YOLO-label codec
(``polyx.core.labels``) as ``class_names``. It holds no label I/O of its
own: the codec is the single owner of the ``.txt`` format and its folder
conventions.
"""
