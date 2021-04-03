## Porting CORVIDS to Python 3

Since Python 2 was sunset on 1 January 2020, it is time for this codebase to be upgraded. I find this tool useful, although hard to use, and wrestling with deprecated software packages doesn't make it easier. So I've decided to fork it and upgrade it.

Although my work history will be contained in the git history, git commit messages do not a plan make. So I will describe my process herein.

## Vision

I would like to see this tool become truly usable without any of the needless overhead of obsolete packages like Tk. I find a UI helpful, but the UI in the latest release does not work with Windows 10 and works poorly on Mac. Moreover, python as a language is ill-equipped for large-scale numerics problems, so in the long run porting this code to pyspark would be beneficial. I'd like to use a web-based front-end, refactor the code to have a usable CLI and API, and finally dockerize it so it can be deployed on modern cloud computing.

## Upgrade Approach

The code is quite clever although was not written with software engineering best practices. That's ok! I can fix that. So here's my upgrade plan from version 2 to version 3:

- Eliminate commented-out and dead code
- Remove the native GUI functionality
- Add integration tests
- Map the functional flow of the software
- Benchmark performance
- Decompose functions into smaller units and add unit tests
- Update to python3
- Update API
- Update CLI
- Wrap with Flask + Dash
- Update documentation
- Port to pyspark

### Upgrade Log

- 04.04.2021: I started this document to track progress and goals. I have started the first few exercises at this point.
  - I removed all dead code and trimmed some whitespace.
  - I deleted the GUI code and removed code references to the GUI functionality in documentation, but not the text.
  - I have added some basic integration tests of the core methods: `recreateData` and `getDataSimple()`. Only the default arguments for `recreateData` are currently tested.
  - I've added integration tests to test some of the arguments for `recreateData`, but they appear not to work.
  - I removed the `analyzeSkew` method that wasn't being used without the GUI code.
  - I removed the unused `getSolutionSpace` method