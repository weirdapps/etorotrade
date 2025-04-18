# Pull Request

## Description
<!-- Provide a concise description of the changes -->

## Type of Change
<!-- Check one or more options that apply -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test coverage improvement

## Quality Checklist
<!-- Make sure all of these are checked before merging -->
- [ ] Code follows project style guidelines
- [ ] Code passes all quality checks (`make lint`)
- [ ] New and existing tests pass (`make test`)
- [ ] Documentation updated (if required)
- [ ] Changes don't reduce code coverage
- [ ] No debugging code or print statements added

## Provider Changes
<!-- If this PR modifies any providers, check the following -->
- [ ] Changes maintain backward compatibility
- [ ] Rate limiting is properly implemented
- [ ] Error handling preserves error hierarchy
- [ ] Data format consistency is maintained
- [ ] Provider interfaces properly implemented
- [ ] No unnecessary API calls added

## Financial Data & Trade Criteria
<!-- If this PR modifies financial data processing or trade criteria -->
- [ ] Trade criteria remain intact
- [ ] Financial calculations verified for accuracy
- [ ] Edge cases and missing data handled properly
- [ ] Format consistency maintained

## Notes
<!-- Any additional information that's important for reviewers -->