Guidelines for Contributing
+++++++++

**Feature Improving**

Contributions to PANINIpy are certainly welcome! Please follow the following guidlines to contribute to the development of the project with new features and modules:

1. **Fork the Repository**: Create a personal fork of the repository on GitHub.

2. **Clone the Fork**: Clone the repository fork to the local machine:

   .. code-block:: console

       git clone https://github.com/[username]/PANINIpy.git

3. **Create a Branch**: Create a new branch for the new feature or bugfix:

   .. code-block:: console

       git checkout -b feature/[new_feature_name]

4. **Make Changes**: Implement changes or additions.

5. **Write Tests**: Write tests for the new code, if applicable.

6. **Commit Changes**: Commit changes with clear and descriptive messages:

   .. code-block:: console

       git commit -a -m "Add feature X"

7. **Push to Fork**: Push the branch to the forked repository:

   .. code-block:: console

       git push origin feature/[new_feature_name]

8. **Submit a Pull Request**: Open a pull request to the `main` branch of the original repository.

   - **Pull Request Template**: Please include the following in the pull request description:

     - A summary of the changes and new feature.
     - Any relevant issue reasoning.
     - A description of any limitations or future work.

9. **Code Review**: Collaborate with the maintainers during the code review process to improve the pull request.

**Issue Reporting**

If anyone encounters any issues or have suggestions for improvements, please open an issue on GitHub. Use the following templates:

- **Bug Report Template**:

  - **Description**: Describe the bug in detail.
  - **Steps to Reproduce**: Provide a step-by-step guide to reproduce the issue.
  - **Expected Behavior**: Explain what you expected to happen.
  - **Actual Behavior**: Describe what actually happened.
  - **Screenshots or Code Snippets**: Include any relevant visuals or code.
  - **System Information**: Operating system, Python version, etc.

- **Feature Request Template**:

  - **Description**: Describe the new feature you would like to see.
  - **Use Case**: Explain why this feature would be useful.
  - **Additional Context**: Any additional information or examples.

**Branch Management**

- The `main` branch contains the latest stable release.
- Feature branches should be created from the `main` branch.
- Please avoid committing directly to the `main` branch.

**Code Style**

- Best to follow PEP 8 style guidelines.
- Use descriptive variable and function names.
- Include documentation strings for all public modules, classes, and functions.