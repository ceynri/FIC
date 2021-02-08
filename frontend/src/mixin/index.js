import Vue from 'vue';
import { mapState, mapMutations } from 'vuex';

import IconBase from '@/components/IconBase.vue';

const mixins = {
  computed: {
    ...mapState(['cssVars']),
  },
  methods: {
    ...mapMutations(['setState']),
  },
};

Vue.mixin(mixins);

Vue.component('IconBase', IconBase);
