import Vue from 'vue';
import { mapState, mapMutations } from 'vuex';

import IconBase from '@/components/IconBase.vue';
import { sizeFormat, percentFormat } from '@/utils/formatter';

const mixins = {
  computed: {
    ...mapState(['cssVars']),
  },
  methods: {
    ...mapMutations(['setState']),
    sizeFormat,
    percentFormat,
  },
};

Vue.mixin(mixins);

Vue.component('IconBase', IconBase);
