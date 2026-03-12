"""
Workflow module custom exceptions
"""


class WorkflowError(Exception):
    """Base exception for workflow errors"""


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found in database"""


class ActivityNotFoundError(WorkflowError):
    """Activity not found in database"""


class WorkflowAlreadyCompletedError(WorkflowError):
    """Workflow already completed, cannot execute"""


class WorkflowFailedError(WorkflowError):
    """Workflow is in failed state"""


class InvalidStateTransitionError(WorkflowError):
    """Invalid workflow or activity state transition"""


class TaskNotFoundError(WorkflowError):
    """Task not found in Celery registry"""


class InvalidWorkflowConfigError(WorkflowError):
    """Invalid workflow configuration"""


class DuplicateTaskNameError(WorkflowError):
    """Task name already exists in Celery registry"""
