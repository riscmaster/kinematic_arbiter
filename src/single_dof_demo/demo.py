# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Demo application showcasing multiple Kalman filter
implementations for signal processing.
"""

import display_gui


def _main():
    demo = display_gui.DisplayGui()
    demo.run()


if __name__ == "__main__":
    _main()
