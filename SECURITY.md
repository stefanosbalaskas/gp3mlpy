# Security policy

Please report security-sensitive issues privately through GitHub's security reporting mechanisms rather than opening a public issue when disclosure could expose users.

`gp3mlpy` deliberately avoids automatic deserialization of arbitrary pickle/joblib artifacts. Optional persisted model artifacts should use audited safe/native formats.
