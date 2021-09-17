"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Helper functions for the database
"""


def get_or_create(session, model, **kwargs):
    """Query or create an instance of a model

    Parameters
    ----------
    session: a SQLAlchemy session
    model: a SQLAlchemy base model
    kwargs: kwargs for the model to be filterd

    Returns
    -------
    instance of the model

    """
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        return instance
