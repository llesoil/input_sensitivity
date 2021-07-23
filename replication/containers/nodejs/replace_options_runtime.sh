#!/bin/sh

$OPTIONS="{flags: ['--jitless','--v8-pool-size=1']}"
`sed -i 's/configs, options = {}/configs, options = $OPTIONS/g' benchmark/common.js`

