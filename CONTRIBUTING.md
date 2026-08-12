# Contributing Guidelines

## Commit Message Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect the meaning of the code (formatting, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or correcting tests
- **chore**: Changes to build process or auxiliary tools

### Scopes (optional)

- `model`: Model loading and inference
- `config`: Configuration files
- `deps`: Dependencies
- `docs`: Documentation
- `ci`: CI/CD configuration

### Examples

```
feat(model): add dynamic expert caching

Implement LRU cache for recently used experts
to reduce disk I/O during inference.
```

```
fix(config): correct repository URL in pyproject.toml

Update placeholder username to actual GitHub user.
```

```
docs(readme): add Stream Experts research focus

Document GGUF quantization and dynamic expert loading
approaches for running large MoE models on CPU.
```

```
chore(deps): simplify dependencies to core requirements

Reduce from 20 to 3 essential packages:
gguf, huggingface-hub, numpy.
```

## Pull Request Process

1. Ensure commits follow the convention above
2. Update documentation as needed
3. Add tests for new features
4. Ensure all tests pass
