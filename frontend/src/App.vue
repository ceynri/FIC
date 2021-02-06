<template>
  <div id="app">
    <TopBar />
    <router-view />
  </div>
</template>

<script>
import TopBar from '@/components/TopBar.vue';

import getKebabCase from '@/utils/getKebabCase';

import '@/styles/base.scss';

export default {
  mounted() {
    this.getCssVars();
  },
  methods: {
    getCssVars() {
      const rootElem = document.documentElement;
      const rootStyle = getComputedStyle(rootElem);
      const cssVars = {};
      const propList = ['primary', 'secondary', 'bg'];
      propList.forEach((prop) => {
        const styleStr = getKebabCase(prop);
        cssVars[prop] = rootStyle.getPropertyValue(`--${styleStr}`).trim();
      });
      this.setState({
        state: 'cssVars',
        value: cssVars,
      });
    },
  },
  components: {
    TopBar,
  },
};
</script>
