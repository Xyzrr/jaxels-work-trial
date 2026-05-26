"""Project automation entry points and shared launcher helpers.

The modules in this package coordinate dataset preparation, model training,
vLLM serving, OpenHands/SWE-bench evaluation, and Kubernetes pod setup. Keep this
package initializer intentionally side-effect free: importing ``scripts`` should
not select a model, dataset, context length, eval stack, or any other ML
experiment setting. Those decisions belong in explicit preset files, CLI
arguments, or the narrowly named module that owns the workflow.
"""
