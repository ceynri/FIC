import Vue from 'vue';

import IconBase from '@/components/IconBase.vue';
import { sizeFormat, percentFormat } from '@/utils/formatter';

const mixins = {
  methods: {
    sizeFormat,
    percentFormat,
  },
};

Vue.mixin(mixins);

Vue.component('IconBase', IconBase);
