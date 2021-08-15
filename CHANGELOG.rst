=========
Changelog
=========

v0.0.3.dev0
===========

New Features
------------

  * Curtailment is now limited to renewable sources. Prior to this
    change, all excess electricity generation would be counted as
    curtailment, which means that in theory, conventional power plants
    could be dispatched to produce excess energy. Now conventional power
    plants are not allowed to produce more than the total demand (modulo
    accounting for storage input and transmission losses, of course)
    while renewable sources can produce excess energy which will be
    accounted for as "curtailment".
    (
      f9be6ce029c5177983ea9c6683f93b1266a2e934
      -> d2c9246e4d69abf3aeef45adc49e44fb19f54fb0
    )

  * Units are now reported without the "/a" and "/h".
    (6db9e90a7b438647f3f4f3891d6fe17eb91915cc)

  * Scalar values equalling zero are no longer reported in the
    corresponding CSV file.
    (
      065f1a666b2ba212ab12b566766993656d1427dc
      85d1d31bdf539c22372e28e6db32d2249377d7e1
    )

  * The "unit"s accompanying "parameter_name"s are now saved when
    reading JSON scenario data.
    (7cb95a7d5d53d62f86065e8764eb17efb1961bad)

  * Temporary files generated to build the results data `Package` are
    now stored in an automatically generated temporary directory. The
    directory can be configured via a command line option in order to
    inspect the generated files, if need be.
    (
      0bb94aa5eda0b3931178c031b882d920cb7ef712
      890a51d967ea81e31e620cff26cf5f632b2d5923
    )

  * The names of exported files can now be configured on the command
    line. At least in a limited manner by specifying their prefix, that
    is.
    (80fe9b3039df479cbd03a32cef86abd3ecc77bd4)

  * The usage of storages and transmission lines can now be controlled
    via specifying penalty costs. These should be small and the
    additional costs incurred through these values will not be reported
    as part of the "variable costs".
    (3dae8aba53cc8ee2834a85c94822879e0633133d)


Fixes
-----

  * Storage investments are now reported under the "parameter name"
    "added capacity". These where previously missing because of an
    excessively restrictive filtering line.
    (bcc7621b1226468d4312b4dc9598664ddeff3ae0)

  * Storage investments should now also work as expected. The parameters
    fixing the relationship between the investments into a storage's
    capacity and its input and output capacity where used in the wrong
    manner and they where applied to the wrong object.
    This leading to all investments being independent of each other and
    input and output capacity investments being unbounded and at no
    costs. Coupling that with another `GenericStorage` quirk lead to
    some really weird issues like storages with no storage capacity
    being used as curtailment.
    Hopefully storage investments now work as expected.
    (
      f761d0c1f9f501e0b935beeb2600eb8262b4892d
      3ac8235716178057fca07fe7d24888816aaa6856
    )

  * Due to a typo, transmission capacities where missing from the values
    reported for the "capacity" "parameter_name" in the exported CSV
    files.
    (34c8f662454b32f50af75b259b7496c0391c7dba)

  * The investment costs where not deducted from the objective value
    when calculating the variable costs leading to reported variable
    costs being too high.
    (84075fca00415c63a355a7efae9adb148f6d142d)



v0.0.2
======

New Features
------------

  * The `--timesteps` option now effectively limits the number of
    time steps that are taken into account for the optimization model.
    (981c6eee29787c6688790b6633726958939a5264)

  * The module's `main` function can now be called like a normal Python
    function again, i.e. it's argument list is no longer modified by
    `click`. So after doing `import mapper as m`, `m.main` now behaves
    as expected.
    In order to achieve this, there's now a `cli` function wrapped by
    `click` to do the command line parsing. This function currently
    immediately calls `main`.
    (c4be52d1c208a403b0a283ef76f5e7140e4423a5)


Fixes
-----

  * The new, more robust `total_demand` calculation didn't work because of
    syntax errors. (8ef0fc09b3c61243e0efacf1f8e892edb9d3992d)

  * The `--year` argument wasn't parsed at all, but simply processed as
    a string. This is wrong, as using it as a key for the parsed input
    data requires the type to be `int`.
    (5922697c9f04ac5c5c1382ba8bd91b07865a8f29)

  * The limits put on e.g. biomass or biogas are now respected properly.
    (
      52940cf8396cfab6ab70a92bda59e2f464e497ab
      15af82a869b116ed6b7812f45268d4180e989290
    )


v0.0.1
======

Initial release. Created before the existence of this changelog, so
there is no feature list. While the tag message claims that this was
used to create the "final" (before curtailment was limited to only apply
to renewable sources) results for data ID43, this is actually wrong,
because there where a few bugs which needed to be fixed. Hence the next
version is the one actually used to generate the results with global
curtailment.
