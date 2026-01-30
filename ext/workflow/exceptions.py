"""
Workflow module custom exceptions
"""


class WorkflowError(Exception):
    """Base exception for workflow errors"""

    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found in database"""

    pass


class ActivityNotFoundError(WorkflowError):
    """Activity not found in database"""

    pass


class WorkflowAlreadyCompletedError(WorkflowError):
    """Workflow already completed, cannot execute"""

    pass


class WorkflowFailedError(WorkflowError):
    """Workflow is in failed state"""

    pass


class InvalidStateTransitionError(WorkflowError):
    """Invalid workflow or activity state transition"""

    pass


class TaskNotFoundError(WorkflowError):
    """Task not found in Celery registry"""

    pass


class InvalidWorkflowConfigError(WorkflowError):
    """Invalid workflow configuration"""

    pass
