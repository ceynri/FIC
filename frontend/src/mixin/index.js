import Vue from 'vue';
import { mapState, mapMutations } from 'vuex';

import IconBase from '@/components/IconBase.vue';

const BTYE_PER_KB = 2 ** 10;
const BTYE_PER_MB = 2 ** 20;

const mixins = {
  computed: {
    ...mapState(['cssVars']),
  },
  methods: {
    ...mapMutations(['setState']),
    /**
     * size大小字符串格式化
     */
    sizeFormat(btye, remainder = 2) {
      if (btye < BTYE_PER_KB) {
        return `${btye} B`;
      }
      if (btye < BTYE_PER_MB) {
        return `${(btye / BTYE_PER_KB).toFixed(remainder)} KB`;
      }
      return `${(btye / BTYE_PER_MB).toFixed(remainder)} MB`;
    },
  },
};

Vue.mixin(mixins);

Vue.component('IconBase', IconBase);
