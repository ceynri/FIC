<template>
  <section class="compressor">
    <PageFrame>
      <header class="title_wrapper">
        <h1 class="title">Compressor</h1>
        <div class="comment">Upload facial images to compress</div>
      </header>
      <Uploader accept="image/*"
    /></PageFrame>
  </section>
</template>

<script>
import PageFrame from '@/components/PageFrame.vue';
import Uploader from '@/components/Uploader.vue';

export default {
  components: {
    PageFrame,
    Uploader,
  },
};
</script>

<style lang="scss" scoped>
.compressor {
  .title {
    font-size: 32px;
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }
}
</style>
